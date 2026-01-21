import ast
import contextlib
import difflib
import io
import os
import re
import shutil
import subprocess
import sys
import time
from functools import lru_cache

import streamlit as st
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group import ContextVariables
from autogen.agentchat.group.patterns import AutoPattern
from autogen.coding import LocalCommandLineCodeExecutor
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from clapp.config import BLABLADOR_MODELS, GEMINI_MODELS, GPT_MODELS, normalize_base_url
from clapp.domain.classy.validation import (
    auto_correct_class_params,
    validate_class_params_in_code,
)
from clapp.llms.providers import build_llm
from clapp.rag.pipeline import retrieve_context


def extract_first_code_block(message: str):
    match = re.search(r"```(?:python)?\n(.*?)```", message, re.DOTALL)
    if match:
        return match.group(1).strip(), True
    return message.strip(), False


def extract_strict_code_block(message: str):
    blocks = re.findall(r"```(?:python)?\n(.*?)```", message, re.DOTALL)
    if len(blocks) != 1:
        return None, False, "Expected a single python code block."
    outside = re.sub(r"```(?:python)?\n.*?```", "", message, flags=re.DOTALL)
    if outside.strip():
        return blocks[0].strip(), False, "Extra text outside the code block."
    return blocks[0].strip(), True, None


def format_code_block(code: str):
    return f"```python\n{code}\n```"


@lru_cache(maxsize=1)
def load_preflight_instructions() -> str:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    prompt_path = os.path.join(base_dir, "prompts", "preflight_instructions.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        return ""
    except Exception:
        return ""


def extract_output_value(code: str):
    patterns = [
        r"['\"]output['\"]\s*:\s*['\"]([^'\"]+)['\"]",
        r"['\"]output['\"]\s*=\s*['\"]([^'\"]+)['\"]",
    ]
    for pattern in patterns:
        match = re.search(pattern, code)
        if match:
            return match.group(1).strip()
    return None


def parse_output_targets(output_value: str) -> set:
    return {token for token in re.split(r"[\s,]+", output_value) if token}


_ALLOWED_COSMO_CALLS = {
    "pk",
    "pk_lin",
    "pk_nonlinear",
    "pk_cb",
    "pk_cb_lin",
    "pk_cb_nonlinear",
    "raw_cl",
    "lensed_cl",
    "lensing_cl",
    "get_transfer",
    "get_transfer_and_k_and_z",
    "get_current_derived_parameters",
    "get_background",
    "get_thermodynamics",
    "get_perturbations",
}


def _span_from_node(node: ast.AST):
    start = getattr(node, "lineno", None)
    if start is None:
        return None
    end = getattr(node, "end_lineno", start)
    return (start - 1, end - 1)


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def find_preflight_spans(code: str) -> list[tuple[int, int]]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    dict_assigns = {}
    spans = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            span = _span_from_node(node.value)
            if span:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        dict_assigns[target.id] = span
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.value, ast.Dict)
            and isinstance(node.target, ast.Name)
        ):
            span = _span_from_node(node.value)
            if span:
                dict_assigns[node.target.id] = span

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        attr = node.func.attr
        if attr == "set":
            if node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Dict):
                    span = _span_from_node(arg)
                    if span:
                        spans.append(span)
                elif isinstance(arg, ast.Name) and arg.id in dict_assigns:
                    spans.append(dict_assigns[arg.id])
            if node.keywords:
                span = _span_from_node(node)
                if span:
                    spans.append(span)
        if attr in _ALLOWED_COSMO_CALLS:
            span = _span_from_node(node)
            if span:
                spans.append(span)

    return merge_spans(spans)


def index_in_spans(index: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= index <= end for start, end in spans)


def changes_within_spans(original: str, updated: str, spans: list[tuple[int, int]]) -> bool:
    if not spans:
        return False
    original_lines = original.splitlines()
    updated_lines = updated.splitlines()
    matcher = difflib.SequenceMatcher(None, original_lines, updated_lines)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag in {"replace", "delete"}:
            for idx in range(i1, i2):
                if not index_in_spans(idx, spans):
                    return False
        if tag in {"replace", "insert"}:
            for idx in range(j1, j2):
                if not index_in_spans(idx, spans):
                    return False
    return True


def contains_param(code: str, key: str) -> bool:
    pattern = rf"['\"]{re.escape(key)}['\"]\s*[:=]"
    return re.search(pattern, code) is not None


def remove_param_entries(code: str, keys: list[str]) -> tuple[str, list[str]]:
    removed = set()
    lines = code.splitlines()
    kept_lines = []
    for line in lines:
        matched = False
        for key in keys:
            if re.search(rf"['\"]{re.escape(key)}['\"]\s*[:=]", line):
                matched = True
                removed.add(key)
        if not matched:
            kept_lines.append(line)
    updated = "\n".join(kept_lines)
    if updated == code:
        updated_inline = code
        for key in keys:
            pattern = rf"(,?\s*['\"]{re.escape(key)}['\"]\s*:\s*[^,}}]+)"
            new_inline = re.sub(pattern, "", updated_inline)
            if new_inline != updated_inline:
                removed.add(key)
            updated_inline = new_inline
        updated_inline = re.sub(r",\s*,", ",", updated_inline)
        updated_inline = re.sub(r"{\s*,", "{", updated_inline)
        updated_inline = re.sub(r",\s*}", "}", updated_inline)
        updated = updated_inline
    return updated, sorted(removed)


def replace_param_key(code: str, old_key: str, new_key: str) -> tuple[str, bool]:
    pattern = r"(['\"])%s\1" % re.escape(old_key)
    updated, count = re.subn(pattern, r"\1%s\1" % new_key, code)
    return updated, count > 0


def replace_param_value_string(
    code: str, key: str, old_values: list[str], new_value: str
) -> tuple[str, bool]:
    changed = False
    for old_value in old_values:
        pattern = (
            r"(['\"]%s['\"]\s*[:=]\s*)['\"]%s['\"]"
            % (re.escape(key), re.escape(old_value))
        )
        code, count = re.subn(pattern, r"\1'%s'" % new_value, code)
        changed = changed or count > 0
    return code, changed


def replace_param_value_bool(code: str, key: str, new_value: str) -> tuple[str, bool]:
    pattern = r"(['\"]%s['\"]\s*[:=]\s*)(True|False|1|0)" % re.escape(key)
    updated, count = re.subn(pattern, r"\1'%s'" % new_value, code)
    return updated, count > 0


def normalize_non_linear_values(code: str) -> tuple[str, list[str], bool]:
    notes = []
    changed = False
    allowed = {"halofit", "hmcode", "none"}
    pattern = r"(['\"]non_linear['\"]\s*[:=]\s*)['\"]([^'\"]+)['\"]"
    matches = list(re.finditer(pattern, code))
    if matches:
        for match in matches:
            value = match.group(2)
            if value.lower() in allowed:
                continue
            replacement = "hmcode" if value.lower() == "mead" else "halofit"
            code = code.replace(match.group(0), f"{match.group(1)}'{replacement}'")
            notes.append(
                f"Normalized non_linear value '{value}' to '{replacement}'."
            )
            changed = True

    list_pattern = (
        r"(nonlinear[_\s]*methods?\s*=\s*\[[^\]]*)['\"]mead['\"]"
    )
    code, list_count = re.subn(list_pattern, r"\1'hmcode'", code, flags=re.IGNORECASE)
    if list_count:
        notes.append("Replaced 'mead' with 'hmcode' in nonlinear method lists.")
        changed = True

    return code, notes, changed


def extract_unread_params(error_output: str) -> list[str]:
    if not error_output:
        return []
    matches = re.findall(
        r"Class did not read input parameter\(s\):\s*([^\n]+)", error_output
    )
    params = []
    for match in matches:
        for token in match.split(","):
            cleaned = token.strip().strip(".")
            if cleaned:
                params.append(cleaned)
    return params


def apply_rule_based_preflight(code: str) -> tuple[str, list[str], bool]:
    notes = []
    changed = False
    output_value = extract_output_value(code)
    pk_keys = ["P_k_max_h/Mpc", "P_k_max_1/Mpc"]
    if output_value:
        outputs = parse_output_targets(output_value)
        if "mPk" not in outputs:
            updated, removed = remove_param_entries(code, pk_keys)
            if removed:
                code = updated
                changed = True
                notes.append(
                    "Removed P_k_max_h/Mpc or P_k_max_1/Mpc because output does not include mPk."
                )
        else:
            if not any(contains_param(code, key) for key in pk_keys):
                notes.append(
                    "Output includes mPk but no P_k_max_1/Mpc or P_k_max_h/Mpc is defined."
                )
    code, renamed_method = replace_param_key(code, "nonlinear_method", "non_linear")
    if renamed_method:
        notes.append("Replaced nonlinear_method with non_linear.")
        changed = True
    code, renamed_flag = replace_param_key(code, "nonlinear", "non_linear")
    if renamed_flag:
        notes.append("Replaced nonlinear with non_linear.")
        changed = True
    code, value_changed = replace_param_value_string(
        code, "non_linear", ["yes", "true", "True", "1"], "halofit"
    )
    if value_changed:
        notes.append("Set non_linear to 'halofit' (CLASS expects halofit/hmcode).")
        changed = True
    code, bool_changed = replace_param_value_bool(code, "non_linear", "halofit")
    if bool_changed:
        notes.append("Set non_linear to 'halofit' (CLASS expects halofit/hmcode).")
        changed = True
    code, mead_changed = replace_param_value_string(
        code, "non_linear", ["mead", "Mead", "MEAD"], "hmcode"
    )
    if mead_changed:
        notes.append("Mapped non_linear 'mead' to 'hmcode'.")
        changed = True
    code, normalized_notes, normalized_changed = normalize_non_linear_values(code)
    if normalized_notes:
        notes.extend(normalized_notes)
    changed = changed or normalized_changed
    return code, notes, changed


def get_preflight_context(code: str, max_chars: int = 3000):
    vector_store = st.session_state.get("vector_store")
    if not vector_store or not code:
        return "", []
    context, evidence = retrieve_context(vector_store, code)
    if context and len(context) > max_chars:
        context = context[:max_chars] + "\n\n[truncated]"
    return context, evidence


def can_run_preflight_llm() -> bool:
    if not st.session_state.get("preflight_enabled", True):
        return False
    if not st.session_state.get("saved_api_key_blablador"):
        return False
    available_models = st.session_state.get("blablador_models") or BLABLADOR_MODELS
    return "alias-fast" in available_models


def build_preflight_prompt(
    code: str,
    rules_text: str,
    doc_context: str,
    rule_notes: list[str],
) -> str:
    rules_block = rules_text or "No additional rules provided."
    docs_block = doc_context or "(No documentation context available.)"
    notes_block = "\n".join(f"- {note}" for note in rule_notes) if rule_notes else "- None"
    return f"""
You are a CLASS preflight checker. Apply the rules below and make minimal changes.
Only edit CLASS parameter dictionaries passed into Class().set or direct CLASS method calls (e.g., cosmo.pk).
Do not modify imports, plotting, or any logic outside CLASS settings or CLASS method call arguments.
Do not reorder keys or remove comments; only add or remove necessary parameter entries or call arguments.

Rules:
{rules_block}

Documentation context:
{docs_block}

Rule-based findings:
{notes_block}

Task:
- If the code already satisfies the rules, reply with exactly: OK
- Otherwise return ONLY one corrected python code block with no extra text
- Keep changes minimal and preserve the user's intent

Python code:
```python
{code}
```
""".strip()


def run_llm_preflight(code: str, rule_notes: list[str]) -> tuple[str, list[str], bool]:
    if not can_run_preflight_llm():
        return code, [], False

    rules_text = load_preflight_instructions()
    doc_context, _evidence = get_preflight_context(code)
    prompt = build_preflight_prompt(code, rules_text, doc_context, rule_notes)

    try:
        preflight_llm = build_llm(
            selected_model="alias-fast",
            api_key=None,
            api_key_gai=None,
            blablador_api_key=st.session_state.saved_api_key_blablador,
            blablador_base_url=st.session_state.blablador_base_url,
            blablador_models=st.session_state.get("blablador_models"),
            streaming=False,
            temperature=0.0,
        )
        messages = [
            SystemMessage(content="Check CLASS parameter consistency for execution."),
            HumanMessage(content=prompt),
        ]
        response = preflight_llm.invoke(messages)
        content = getattr(response, "content", "")
    except Exception as exc:
        return code, [f"Preflight model failed: {exc}"], False

    if content.strip().upper() == "OK":
        return code, [], False

    updated_code, is_strict, _strict_error = extract_strict_code_block(content)
    if not is_strict or not updated_code:
        updated_code, has_code = extract_first_code_block(content)
        if not has_code:
            return code, ["Preflight model returned non-code output; ignored."], False

    if updated_code.strip() == code.strip():
        return code, [], False

    spans = find_preflight_spans(code)
    if not spans:
        return (
            code,
            [
                "Preflight could not locate CLASS settings or CLASS method calls; edits ignored."
            ],
            False,
        )
    if not changes_within_spans(code, updated_code, spans):
        return (
            code,
            [
                "Preflight model changes outside CLASS settings or CLASS method calls were ignored."
            ],
            False,
        )

    return (
        updated_code,
        ["Preflight model updated the code using alias-fast."],
        True,
    )


def run_preflight_checks(code: str) -> tuple[str, list[str], bool]:
    if not st.session_state.get("preflight_enabled", True):
        return code, [], False

    notes = []
    changed = False

    code, rule_notes, rule_changed = apply_rule_based_preflight(code)
    notes.extend(rule_notes)
    changed = changed or rule_changed

    code, llm_notes, llm_changed = run_llm_preflight(code, rule_notes)
    notes.extend(llm_notes)
    changed = changed or llm_changed

    return code, notes, changed


def summarize_execution_output(output: str, max_lines: int = 60, max_chars: int = 4000):
    if not output:
        return ""
    lines = output.strip().splitlines()
    traceback_start = None
    for idx, line in enumerate(lines):
        if "Traceback (most recent call last)" in line:
            traceback_start = idx
    if traceback_start is not None:
        lines = lines[traceback_start:]
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    summary = "\n".join(lines)
    if len(summary) > max_chars:
        summary = summary[-max_chars:]
    return summary


def get_error_context(error_summary: str, max_chars: int = 3000):
    vector_store = st.session_state.get("vector_store")
    if not vector_store or not error_summary:
        return "", []
    context, evidence = retrieve_context(vector_store, error_summary)
    if context and len(context) > max_chars:
        context = context[:max_chars] + "\n\n[truncated]"
    return context, evidence


class PlotAwareExecutor(LocalCommandLineCodeExecutor):
    def __init__(self, **kwargs):
        import tempfile

        plots_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "plots"
        )
        os.makedirs(plots_dir, exist_ok=True)

        temp_dir = tempfile.TemporaryDirectory()
        kwargs["work_dir"] = temp_dir.name
        super().__init__(**kwargs)
        self._temp_dir = temp_dir
        self._plots_dir = plots_dir

    @contextlib.contextmanager
    def _capture_output(self):
        old_out, old_err = sys.stdout, sys.stderr
        buf_out, buf_err = io.StringIO(), io.StringIO()
        sys.stdout, sys.stderr = buf_out, buf_err
        try:
            yield buf_out, buf_err
        finally:
            sys.stdout, sys.stderr = old_out, old_err

    def execute_code(self, code: str):
        match = re.search(r"```(?:python)?\n(.*?)```", code, re.DOTALL)
        cleaned = match.group(1) if match else code
        cleaned = cleaned.replace("plt.show()", "")

        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        plot_filename = f"plot_{timestamp}.png"
        plot_path = os.path.join(self._plots_dir, plot_filename)
        temp_plot_path = None

        for line in cleaned.split("\n"):
            if "plt.savefig" in line:
                leading_spaces = line[: len(line) - len(line.lstrip())]
                temp_plot_path = os.path.join(
                    self._temp_dir.name, f"temporary_{timestamp}.png"
                )
                new_line = f"{leading_spaces}plt.savefig('{temp_plot_path}', dpi=300)"
                cleaned = cleaned.replace(line, new_line)
                break

        temp_script_path = os.path.join(
            self._temp_dir.name, f"temp_script_{timestamp}.py"
        )
        saved_script_path = os.path.join(self._plots_dir, f"executed_{timestamp}.py")
        with open(temp_script_path, "w", encoding="utf-8") as file:
            file.write(cleaned)
        try:
            shutil.copy2(temp_script_path, saved_script_path)
        except OSError:
            saved_script_path = None

        full_output = ""
        try:
            process = subprocess.Popen(
                [sys.executable, temp_script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=self._temp_dir.name,
            )
            stdout, _ = process.communicate()

            with self._capture_output() as (out_buf, err_buf):
                if stdout:
                    out_buf.write(stdout)
                stdout_text = out_buf.getvalue()
                stderr_text = err_buf.getvalue()

            if stdout_text:
                full_output += f"STDOUT:\n{stdout_text}\n"
            if stderr_text:
                full_output += f"STDERR:\n{stderr_text}\n"

            if temp_plot_path and os.path.exists(temp_plot_path):
                shutil.copy2(temp_plot_path, plot_path)
                if "generated_plots" not in st.session_state:
                    st.session_state.generated_plots = []
                st.session_state.generated_plots.append(plot_path)

            if saved_script_path:
                full_output += f"Saved script: {saved_script_path}\n"

        except Exception:
            with self._capture_output() as (_out_buf, err_buf):
                import traceback

                traceback.print_exc(file=sys.stderr)
                full_output += f"STDERR:\n{err_buf.getvalue()}\n"

        return full_output, plot_path


EXECUTOR = PlotAwareExecutor(timeout=10)


def get_autofix_llm(
    selected_model, api_key, api_key_gai, blablador_api_key, blablador_base_url
):
    blablador_models = st.session_state.get("blablador_models") or BLABLADOR_MODELS
    if selected_model in blablador_models:
        if not blablador_api_key:
            return None, "Missing Blablador API key."
        llm = ChatOpenAI(
            model_name=selected_model,
            openai_api_key=blablador_api_key,
            base_url=normalize_base_url(blablador_base_url),
            temperature=0.2,
        )
        return llm, None
    if selected_model in GEMINI_MODELS:
        if not api_key_gai:
            return None, "Missing Gemini API key."
        llm = ChatGoogleGenerativeAI(
            model=selected_model,
            google_api_key=api_key_gai,
            temperature=0.2,
            convert_system_message_to_human=True,
        )
        return llm, None
    if selected_model in GPT_MODELS:
        if not api_key:
            return None, "Missing OpenAI API key."
        llm = ChatOpenAI(
            model_name=selected_model,
            openai_api_key=api_key,
            temperature=0.2,
        )
        return llm, None
    return None, "Unsupported model for auto-fix."


def get_alias_large_second_opinion(
    error_summary: str,
    code: str,
    docs_block: str,
    removed_params_block: str,
    preflight_notes_block: str,
    blablador_api_key: str | None,
    blablador_base_url: str,
    selected_model: str | None,
):
    blablador_models = st.session_state.get("blablador_models") or BLABLADOR_MODELS
    if selected_model == "alias-large":
        return ""
    if "alias-large" not in blablador_models or not blablador_api_key:
        return ""
    try:
        advisor = build_llm(
            selected_model="alias-large",
            api_key=None,
            api_key_gai=None,
            blablador_api_key=blablador_api_key,
            blablador_base_url=blablador_base_url,
            blablador_models=blablador_models,
            streaming=False,
            temperature=0.0,
        )
        review_prompt = f"""
You are a senior Python reviewer. Analyze the error and suggest minimal fixes.
Return only short bullet points. Do NOT output code.

Error summary:
{error_summary}
{docs_block}
{removed_params_block}
{preflight_notes_block}

Code:
```python
{code}
```
""".strip()
        response = advisor.invoke(
            [
                SystemMessage(content="Provide concise fix suggestions."),
                HumanMessage(content=review_prompt),
            ]
        )
        content = getattr(response, "content", "")
        return content.strip() if content else ""
    except Exception:
        return ""


def auto_fix_with_direct_llm(
    review_message,
    selected_model,
    api_key,
    api_key_gai,
    blablador_api_key,
    blablador_base_url,
):
    llm, error = get_autofix_llm(
        selected_model,
        api_key,
        api_key_gai,
        blablador_api_key,
        blablador_base_url,
    )
    if error:
        return None, error
    try:
        response = llm.invoke([HumanMessage(content=review_message)])
        content = getattr(response, "content", None)
        return content if content else str(response), None
    except Exception as exc:
        return None, str(exc)


def call_code(
    agents,
    last_assistant_message,
    selected_model,
    api_key,
    api_key_gai,
    blablador_api_key,
    blablador_base_url,
):
    if last_assistant_message:
        code_to_execute, _has_code_block = extract_first_code_block(
            last_assistant_message
        )
        if not code_to_execute:
            return None, "No code found to execute in the previous messages.", ""
        st.markdown("Executing code...")
        st.info("Executing cleaned code...")
        invalid_params, suggestions = validate_class_params_in_code(code_to_execute)
        if invalid_params:
            code_to_execute, replacements = auto_correct_class_params(
                code_to_execute, suggestions
            )
            if replacements:
                st.info(
                    "Auto-corrected parameters: "
                    + ", ".join([f"{old} -> {new}" for old, new in replacements])
                )
            invalid_params, suggestions = validate_class_params_in_code(code_to_execute)
        if invalid_params:
            suggestions_text = ""
            for param in invalid_params:
                if suggestions.get(param):
                    suggestions_text += (
                        f"\n  - {param}: {', '.join(suggestions[param])}"
                    )
            st.warning(
                "Invalid CLASS input parameter(s) detected before execution: "
                f"{', '.join(invalid_params)}"
            )
            if suggestions_text:
                st.info(f"Suggestions:{suggestions_text}")
        code_to_execute, preflight_notes, preflight_changed = run_preflight_checks(
            code_to_execute
        )
        if preflight_notes:
            note_text = " ".join(preflight_notes)
            if preflight_changed:
                st.warning(f"Preflight adjustments: {note_text}")
            else:
                st.info(f"Preflight notes: {note_text}")
        execution_output, plot_path = EXECUTOR.execute_code(code_to_execute)
        st.subheader("Execution Output")
        st.text(execution_output)

        if os.path.exists(plot_path):
            st.success("Plot generated successfully!")
            st.image(plot_path, width=700)
        else:
            st.warning("No plot was generated")

        has_errors = any(
            error_indicator in execution_output
            for error_indicator in [
                "Traceback",
                "Error:",
                "Exception:",
                "TypeError:",
                "ValueError:",
                "NameError:",
                "SyntaxError:",
                "Error in Class",
            ]
        )

        max_iterations = 3
        current_iteration = 0

        attempted_fixes = False
        auto_fix_available = bool(agents)
        while has_errors and current_iteration < max_iterations:
            if not auto_fix_available:
                st.warning(
                    "Automatic error-fixing is not available for this model. Please edit the code and try again."
                )
                break
            attempted_fixes = True
            current_iteration += 1
            st.error(f"Previous error: {execution_output}")
            st.info(f"Fixing errors (attempt {current_iteration}/{max_iterations})...")

            removed_params_block = ""
            unread_params = extract_unread_params(execution_output)
            if unread_params:
                code_to_execute, removed_params = remove_param_entries(
                    code_to_execute, unread_params
                )
                if removed_params:
                    removed_text = ", ".join(removed_params)
                    st.warning(
                        "Removed parameters rejected by CLASS: " + removed_text
                    )
                    removed_params_block = (
                        "Removed parameters rejected by CLASS: " + removed_text + "\n"
                    )

            preflight_notes_block = ""
            preflight_code, preflight_notes, preflight_changed = run_preflight_checks(
                code_to_execute
            )
            if preflight_notes:
                note_text = " ".join(preflight_notes)
                if preflight_changed:
                    st.warning(f"Preflight review adjustments: {note_text}")
                else:
                    st.info(f"Preflight review notes: {note_text}")
                preflight_notes_block = f"Preflight review notes:\n{note_text}\n"
            if preflight_changed:
                code_to_execute = preflight_code

            error_summary = summarize_execution_output(execution_output)
            doc_context, error_evidence = get_error_context(error_summary)
            if error_evidence:
                st.session_state["last_error_evidence"] = error_evidence
            docs_block = (
                f"\n\nRelevant documentation:\n{doc_context}\n"
                if doc_context
                else ""
            )
            second_opinion = get_alias_large_second_opinion(
                error_summary=error_summary or execution_output,
                code=code_to_execute,
                docs_block=docs_block,
                removed_params_block=removed_params_block,
                preflight_notes_block=preflight_notes_block,
                blablador_api_key=blablador_api_key,
                blablador_base_url=blablador_base_url,
                selected_model=selected_model,
            )
            second_opinion_block = (
                f"\n\nSecond opinion (alias-large):\n{second_opinion}\n"
                if second_opinion
                else ""
            )
            if second_opinion:
                st.info("Second opinion (alias-large) added to auto-fix prompt.")
            review_message = f"""
            Previous answer had errors during execution:
            {error_summary or execution_output}
            {docs_block}
            {removed_params_block}
            {preflight_notes_block}
            {second_opinion_block}
            Please fix the code to resolve the errors. Return ONLY one corrected python code block, no explanations or extra text.
            {format_code_block(code_to_execute)}
            """

            shared_context = ContextVariables(
                data={
                    "user_prompt": "Correct the errors in the code",
                    "last_answer": last_assistant_message,
                    "feedback": f" Previous answer had errors during execution: {error_summary or execution_output}",
                    "rating": 0,
                    "revisions": 0,
                }
            )

            if selected_model in GEMINI_MODELS:
                pattern = AutoPattern(
                    initial_agent=agents["refine_agent_gai"],
                    agents=[agents["refine_agent_gai"]],
                    group_manager_args={"llm_config": agents["initial_config_gai"]},
                    context_variables=shared_context,
                )
            else:
                pattern = AutoPattern(
                    initial_agent=agents["refine_agent_final"],
                    agents=[agents["refine_agent_final"]],
                    group_manager_args={"llm_config": agents["initial_config"]},
                    context_variables=shared_context,
                )

            formatted_answer = None
            try:
                result, _context_variables, _last_agent = initiate_group_chat(
                    pattern=pattern,
                    messages=review_message,
                    max_rounds=2,
                )
                formatted_answer = result.chat_history[-1]["content"]
            except Exception as exc:
                if st.session_state.debug:
                    st.session_state.debug_messages.append(
                        ("Auto-fix group chat error", str(exc))
                    )
                formatted_answer, autofix_error = auto_fix_with_direct_llm(
                    review_message,
                    selected_model,
                    api_key,
                    api_key_gai,
                    blablador_api_key,
                    blablador_base_url,
                )
                if autofix_error:
                    st.error(f"Auto-fix failed: {autofix_error}")
                    break

            if st.session_state.debug:
                st.session_state.debug_messages.append(
                    ("Error Review Feedback", formatted_answer)
                )

            if not formatted_answer:
                st.error("Auto-fix did not return a response.")
                break

            code_to_execute, is_strict, strict_error = extract_strict_code_block(
                formatted_answer
            )
            if not is_strict:
                if st.session_state.debug and strict_error:
                    st.session_state.debug_messages.append(
                        ("Auto-fix format issue", strict_error)
                    )
                strict_message = f"""
                Return ONLY one python code block, with no extra text or explanations.
                Do not change the code; just output the corrected code block from this response:
                {formatted_answer}
                """
                reformatted_answer, reform_error = auto_fix_with_direct_llm(
                    strict_message,
                    selected_model,
                    api_key,
                    api_key_gai,
                    blablador_api_key,
                    blablador_base_url,
                )
                if reform_error:
                    st.error(f"Auto-fix formatting failed: {reform_error}")
                    break
                formatted_answer = reformatted_answer
                code_to_execute, is_strict, strict_error = extract_strict_code_block(
                    formatted_answer
                )
                if not is_strict:
                    st.error("Auto-fix response was not code-only.")
                    break

            st.info("Executing corrected code...")
            invalid_params, suggestions = validate_class_params_in_code(code_to_execute)
            if invalid_params:
                code_to_execute, replacements = auto_correct_class_params(
                    code_to_execute, suggestions
                )
                if replacements:
                    st.info(
                        "Auto-corrected parameters: "
                        + ", ".join([f"{old} -> {new}" for old, new in replacements])
                    )
            invalid_params, suggestions = validate_class_params_in_code(
                code_to_execute
            )
            if invalid_params:
                suggestions_text = ""
                for param in invalid_params:
                    if suggestions.get(param):
                        suggestions_text += (
                            f"\n  - {param}: {', '.join(suggestions[param])}"
                        )
                st.warning(
                    "Invalid CLASS input parameter(s) detected before execution: "
                    f"{', '.join(invalid_params)}"
                )
                if suggestions_text:
                    st.info(f"Suggestions:{suggestions_text}")
            code_to_execute, preflight_notes, preflight_changed = run_preflight_checks(
                code_to_execute
            )
            if preflight_notes:
                note_text = " ".join(preflight_notes)
                if preflight_changed:
                    st.warning(f"Preflight adjustments: {note_text}")
                else:
                    st.info(f"Preflight notes: {note_text}")
            execution_output, plot_path = EXECUTOR.execute_code(code_to_execute)
            st.subheader("Execution Output")
            st.text(execution_output)

            if os.path.exists(plot_path):
                st.success("Plot generated successfully!")
                st.image(plot_path, width=700)
            else:
                st.warning("No plot was generated")

            if st.session_state.debug:
                st.session_state.debug_messages.append(
                    ("Execution Output", execution_output)
                )

            last_assistant_message = formatted_answer
            has_errors = any(
                error_indicator in execution_output
                for error_indicator in [
                    "Traceback",
                    "Error:",
                    "Exception:",
                    "TypeError:",
                    "ValueError:",
                    "NameError:",
                    "SyntaxError:",
                    "Error in Class",
                ]
            )

        if has_errors:
            if attempted_fixes:
                st.markdown(
                    "> **Note**: Some errors could not be fixed after multiple attempts. "
                    "You can request changes by describing them in the chat."
                )
            else:
                st.markdown(
                    "> **Note**: Auto-fix is unavailable for this model. Please correct the code and try again."
                )
            st.markdown(f"> Last execution message:\n{execution_output}")

            with st.expander("View Failed Code", expanded=False):
                st.markdown(format_code_block(code_to_execute))
            response_text = (
                f"Execution completed with errors:\n{execution_output}\n\n"
                f"The following code was executed:\n{format_code_block(code_to_execute)}\n"
            )
        else:
            if any(
                error_indicator in execution_output
                for error_indicator in [
                    "Traceback",
                    "Error:",
                    "Exception:",
                    "TypeError:",
                    "ValueError:",
                    "NameError:",
                    "SyntaxError:",
                ]
            ):
                st.markdown(
                    "> **Note**: Code execution completed but with errors. "
                    "You can request changes by describing them in the chat."
                )
                st.markdown(f"> Execution message:\n{execution_output}")

                with st.expander("View Failed Code", expanded=False):
                    st.markdown(format_code_block(code_to_execute))
                response_text = (
                    f"Execution completed with errors:\n{execution_output}\n\n"
                    f"The following code was executed:\n{format_code_block(code_to_execute)}\n"
                )
            else:
                st.markdown(
                    "> Code executed successfully. Last execution message:\n"
                    f"{execution_output}"
                )

                with st.expander("View Successfully Executed Code", expanded=False):
                    st.markdown(format_code_block(code_to_execute))

                response_text = (
                    f"Execution completed successfully:\n{execution_output}\n\n"
                    f"The following code was executed:\n{format_code_block(code_to_execute)}"
                )

                if os.path.exists(plot_path):
                    response_text += f"\n\nPLOT_PATH:{plot_path}\n"
        return response_text, execution_output, plot_path
    return "No code found to execute in the previous messages.", "", ""
