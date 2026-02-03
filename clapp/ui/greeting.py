import streamlit as st
import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

from clapp.config import GEMINI_MODELS
from clapp.llms.providers import build_llm
from clapp.ui.streaming import StreamHandler
from clapp.utils.llm_errors import format_llm_error, is_model_temporarily_unavailable


def _is_blablador_model(model_id: str | None) -> bool:
    if not model_id:
        return False
    if model_id.startswith("alias-"):
        return True
    return model_id in (st.session_state.get("blablador_models") or [])


def _build_blablador_greeting_chain() -> list[str]:
    meta = st.session_state.get("blablador_models_meta", {})
    detected = set(st.session_state.get("blablador_models") or [])

    def ok(model_id: str) -> bool:
        if model_id not in detected:
            return False
        try:
            return int(meta.get(model_id, 0)) > 0
        except Exception:
            return False

    chain = []
    current_best = st.session_state.get("current_best_minimax_id")
    if current_best and ok(current_best):
        chain.append(current_best)
    for candidate in ("alias-huge", "alias-large", "alias-code", "alias-fast"):
        if ok(candidate):
            chain.append(candidate)
    return chain


def _label_for_model(model_id: str) -> str:
    current_best = st.session_state.get("current_best_minimax_id")
    if current_best and model_id == current_best:
        return "current-best-MiniMax"
    return model_id


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

            selected_model = st.session_state.selected_model
            models_to_try = [selected_model] if selected_model else []
            if _is_blablador_model(selected_model):
                models_to_try = _build_blablador_greeting_chain()

            if not models_to_try:
                st.error(
                    "No available model found for greeting. Please select a model from the sidebar."
                )
                st.session_state.greeted = True
                st.rerun()
                return

            messages = [
                SystemMessage(content=initial_instructions),
                HumanMessage(
                    content=(
                        "Please greet the user and briefly explain what you can do as the CLASS code assistant."
                    )
                ),
            ]

            greeting = None
            last_exc = None
            for idx, model_id in enumerate(models_to_try, start=1):
                welcome_container.markdown(
                    f"Using `{_label_for_model(model_id)}` for greeting ({idx}/{len(models_to_try)})..."
                )
                streaming_llm = build_llm(
                    model_id,
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
                try:
                    greeting = streaming_llm.invoke(messages)
                    last_exc = None
                    if st.session_state.selected_model != model_id:
                        st.session_state.selected_model = model_id
                        st.session_state.previous_model = model_id
                    break
                except Exception as exc:
                    last_exc = exc
                    if idx < len(models_to_try) and is_model_temporarily_unavailable(
                        exc
                    ):
                        continue
                    if idx < len(models_to_try) and _is_blablador_model(model_id):
                        continue
                    break

            if greeting is None:
                error_message = (
                    format_llm_error(last_exc) if last_exc else "Greeting failed."
                )
                if _is_blablador_model(selected_model) and (
                    st.session_state.get("blablador_models") or []
                ):
                    error_message = (
                        "Blablador is responding, but the selected models are currently unavailable. "
                        "Please choose another model from the dropdown in the sidebar."
                    )
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )
                st.session_state.memory.add_ai_message(error_message)
                st.session_state.greeted = True
                if st.session_state.get("debug") and last_exc:
                    st.session_state.debug_messages.append(
                        ("Greeting error", str(last_exc))
                    )
                st.rerun()
                return

            st.session_state.messages.append(
                {"role": "assistant", "content": greeting.content}
            )
            st.session_state.memory.add_ai_message(greeting.content)
            st.session_state.greeted = True

            if st.session_state.selected_model in GEMINI_MODELS:
                st.markdown(greeting.content)

            st.rerun()
            return
