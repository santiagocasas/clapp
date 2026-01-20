import contextlib
import io
import os
import re
import subprocess
import sys
import time

import streamlit as st
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group import ContextVariables
from autogen.agentchat.group.patterns import AutoPattern
from autogen.coding import LocalCommandLineCodeExecutor
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from clapp.config import BLABLADOR_MODELS, GEMINI_MODELS, GPT_MODELS, normalize_base_url
from clapp.domain.classy.validation import (
    auto_correct_class_params,
    validate_class_params_in_code,
)


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
        with open(temp_script_path, "w", encoding="utf-8") as file:
            file.write(cleaned)

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
                import shutil

                shutil.copy2(temp_plot_path, plot_path)
                if "generated_plots" not in st.session_state:
                    st.session_state.generated_plots = []
                st.session_state.generated_plots.append(plot_path)

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
    if selected_model in BLABLADOR_MODELS:
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
            execution_output = (
                "STDOUT:\n"
                "Error: Invalid CLASS input parameter(s) detected before execution:\n"
                f"{', '.join(invalid_params)}\n"
            )
            if suggestions_text:
                execution_output += f"\nSuggestions:{suggestions_text}\n"
            plot_path = ""
        else:
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

            review_message = f"""
            Previous answer had errors during execution:
            {execution_output}

            Please fix the code to resolve the errors. Return ONLY one corrected python code block, no explanations or extra text.
            {format_code_block(code_to_execute)}
            """

            shared_context = ContextVariables(
                data={
                    "user_prompt": "Correct the errors in the code",
                    "last_answer": last_assistant_message,
                    "feedback": f" Previous answer had errors during execution: {execution_output}",
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
                execution_output = (
                    "STDOUT:\n"
                    "Error: Invalid CLASS input parameter(s) detected before execution:\n"
                    f"{', '.join(invalid_params)}\n"
                )
                if suggestions_text:
                    execution_output += f"\nSuggestions:{suggestions_text}\n"
                plot_path = ""
            else:
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
