import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

from clapp.config import GEMINI_MODELS
from clapp.llms.providers import build_llm
from clapp.ui.streaming import StreamHandler
from clapp.utils.llm_errors import format_llm_error


def maybe_greet(initial_instructions, api_key, api_key_gai):
    if (
        "llm_initialized" in st.session_state
        and st.session_state.llm_initialized
        and st.session_state.vector_store
        and not st.session_state.greeted
    ):
        with st.chat_message("assistant"):
            welcome_container = st.empty()
            welcome_stream_handler = StreamHandler(welcome_container)

            streaming_llm = build_llm(
                st.session_state.selected_model,
                api_key,
                api_key_gai,
                st.session_state.saved_api_key_blablador,
                st.session_state.blablador_base_url,
                st.session_state.get("blablador_models"),
                callbacks=[welcome_stream_handler],
                streaming=True,
                temperature=1.0,
                timeout=10,
            )

            messages = [
                SystemMessage(content=initial_instructions),
                HumanMessage(
                    content=(
                        "Please greet the user and briefly explain what you can do as the CLASS code assistant."
                    )
                ),
            ]

            try:
                greeting = streaming_llm.invoke(messages)
            except Exception as exc:
                error_message = format_llm_error(exc)
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )
                st.session_state.memory.add_ai_message(error_message)
                st.session_state.greeted = True
                if st.session_state.get("debug"):
                    st.session_state.debug_messages.append(("Greeting error", str(exc)))
                return

            st.session_state.messages.append(
                {"role": "assistant", "content": greeting.content}
            )
            st.session_state.memory.add_ai_message(greeting.content)
            st.session_state.greeted = True

            if st.session_state.selected_model in GEMINI_MODELS:
                st.markdown(greeting.content)
