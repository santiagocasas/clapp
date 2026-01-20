import os

import os

import streamlit as st

from clapp.config import GEMINI_MODELS
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


def render_chat(options, api_key, api_key_gai, initial_instructions):
    has_any_key = bool(
        api_key or api_key_gai or st.session_state.get("saved_api_key_blablador")
    )

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

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.memory.add_user_message(user_input)
        context, evidence = retrieve_context(
            st.session_state.vector_store, user_input
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
                    callbacks=[stream_handler],
                    streaming=True,
                    temperature=0.2,
                )

            if user_input.strip().lower() in {"execute!", "plot!"}:
                response = run_code_request()
            else:
                response = call_ai(context, user_input, initial_instructions)
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
