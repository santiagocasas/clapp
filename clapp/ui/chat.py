import json
import os
from difflib import SequenceMatcher

import streamlit as st

from clapp.llms.providers import build_llm
from clapp.rag.pipeline import retrieve_context
from clapp.services.orchestrator import call_ai, run_code_request
from clapp.ui.streaming import StreamHandler


def estimate_token_count(text, model_name):
    if not text:
        return 0
    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model_name or "gpt-4")
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return len(text.split())


EXAMPLE_PROMPTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "examples",
    "example_prompts.json",
)
EXAMPLE_SIMILARITY_THRESHOLD = 0.9


def load_example_prompts():
    try:
        with open(EXAMPLE_PROMPTS_PATH, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    examples = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        prompt = entry.get("prompt")
        code = entry.get("code")
        example_id = entry.get("id")
        if not prompt or not code:
            continue
        examples.append({"id": example_id, "prompt": prompt, "code": code})
    return examples


def find_example_match(prompt, examples):
    prompt = prompt.strip()
    if not prompt:
        return None
    for entry in examples:
        if entry["prompt"].strip() == prompt:
            return entry
    best_match = None
    best_ratio = EXAMPLE_SIMILARITY_THRESHOLD
    for entry in examples:
        ratio = SequenceMatcher(None, prompt.lower(), entry["prompt"].lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = entry
    return best_match


def handle_example_prompt():
    selected = st.session_state.get("example_prompt_pills")
    if selected:
        st.session_state["queued_prompt"] = selected
        st.session_state["example_prompt_pills"] = None


def render_chat(options, api_key, api_key_gai, initial_instructions):
    has_any_key = bool(
        api_key or api_key_gai or st.session_state.get("saved_api_key_blablador")
    )
    example_prompts = load_example_prompts()

    if options and st.session_state.vector_store:
        user_input = st.chat_input("Type your prompt here...")
    else:
        if not has_any_key:
            st.markdown(
                """
                <div style="text-align: center; font-size: 1.5rem; font-weight: 600; margin-top: 1rem;">
                    Please enter an API key to use the app
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif not st.session_state.vector_store:
            st.markdown(
                """
                <div style="text-align: center; font-size: 1.5rem; font-weight: 600; margin-top: 1rem;">
                    Please generate an embedding before using the app
                </div>
                """,
                unsafe_allow_html=True,
            )
        user_input = None

    if "show_example_prompts" not in st.session_state:
        st.session_state.show_example_prompts = True

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "PLOT_PATH:" in message["content"]:
                parts = message["content"].split("PLOT_PATH:")
                st.markdown(parts[0])
                for plot_info in parts[1:]:
                    plot_path = plot_info.split("\n")[0].strip()
                    if os.path.exists(plot_path):
                        st.image(plot_path, width=700)
            else:
                st.markdown(message["content"])

    if st.session_state.get("greeted") and options and st.session_state.vector_store:
        show_examples = st.toggle(
            "Show example prompts",
            key="show_example_prompts",
        )
        if show_examples and example_prompts:
            st.caption("Try one of these example questions:")
            st.pills(
                "Example prompts",
                [entry["prompt"] for entry in example_prompts],
                selection_mode="single",
                label_visibility="collapsed",
                key="example_prompt_pills",
                on_change=handle_example_prompt,
            )

    if not user_input and options and st.session_state.vector_store:
        queued_prompt = st.session_state.pop("queued_prompt", None)
        if queued_prompt:
            user_input = queued_prompt

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        canonical_match = find_example_match(user_input, example_prompts)
        canonical_prompt = user_input
        if canonical_match:
            canonical_prompt = (
                "Canonical reference solution (use this as the template and keep changes minimal). "
                "Return only one python code block.\n\n"
                f"```python\n{canonical_match['code']}\n```\n\n"
                f"User request: {user_input}"
            )

        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.memory.add_user_message(user_input)
        context, evidence = retrieve_context(
            st.session_state.vector_store, canonical_prompt
        )
        st.session_state["last_context"] = context
        st.session_state["last_evidence"] = evidence

        with st.chat_message("assistant"):
            stream_box = st.empty()
            stream_handler = StreamHandler(stream_box)

            if st.session_state.mode_is_fast == "Fast Mode":
                st.session_state.llm = build_llm(
                    st.session_state.selected_model,
                    api_key,
                    api_key_gai,
                    st.session_state.saved_api_key_blablador,
                    st.session_state.blablador_base_url,
                    st.session_state.get("blablador_models"),
                    callbacks=[stream_handler],
                    streaming=True,
                    temperature=0.2,
                )

            if user_input.strip().lower() in {"execute!", "plot!", "run!"}:
                response = run_code_request()
            else:
                response = call_ai(context, canonical_prompt, initial_instructions)
                if st.session_state.mode_is_fast != "Fast Mode":
                    st.markdown(response.content)

            st.session_state.last_token_count = estimate_token_count(
                response.content, st.session_state.selected_model
            )
            st.session_state.memory.add_ai_message(response.content)
            st.session_state.messages.append(
                {"role": "assistant", "content": response.content}
            )
            if "```" in response.content:
                st.rerun()
