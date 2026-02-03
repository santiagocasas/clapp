import streamlit as st

from clapp.agents.groupchat import build_groupchat_pattern, get_agents, run_group_chat
from clapp.config import BLABLADOR_MODELS
from clapp.execution.code_execution import call_code as run_code
from clapp.rag.pipeline import build_messages, format_memory_messages
from clapp.utils.llm_errors import format_llm_error


class Response:
    def __init__(self, content):
        self.content = content


def _is_blablador_model(model_id: str | None) -> bool:
    if not model_id:
        return False
    if model_id in BLABLADOR_MODELS or model_id.startswith("alias-"):
        return True
    return model_id in (st.session_state.get("blablador_models") or [])


def run_code_request():
    agents = get_agents()
    last_assistant_message = None
    for message in reversed(st.session_state.messages):
        if message["role"] == "assistant" and "```" in message["content"]:
            last_assistant_message = message["content"]
            break

    response_text, _execution_output, _plot_path = run_code(
        agents,
        last_assistant_message,
        st.session_state.selected_model,
        st.session_state.get("saved_api_key"),
        st.session_state.get("saved_api_key_gai"),
        st.session_state.get("saved_api_key_blablador"),
        st.session_state.blablador_base_url,
    )
    return Response(content=response_text)


def call_ai(context, user_input, initial_instructions):
    agents = get_agents()
    if _is_blablador_model(st.session_state.selected_model):
        if st.session_state.mode_is_fast != "Fast Mode":
            return Response(content="Blablador models are only available in Fast Mode.")
    if st.session_state.mode_is_fast == "Fast Mode":
        messages = build_messages(
            context,
            user_input,
            initial_instructions,
            st.session_state.memory.messages,
        )
        response = []
        try:
            for chunk in st.session_state.llm.stream(messages):
                response.append(chunk.content)
        except Exception as exc:
            error_message = format_llm_error(exc)
            if st.session_state.get("debug"):
                st.session_state.debug_messages.append(("Streaming error", str(exc)))
            return Response(content=error_message)

        response = "".join(response)
        return Response(content=response)

    st.markdown("Thinking (Deep Thought Mode)... ")
    conversation_history = format_memory_messages(st.session_state.memory.messages)
    shared_context = {
        "user_prompt": user_input,
        "last_answer": "see chat history",
        "feedback": "see chat history",
        "rating": 0,
        "revisions": 0,
    }
    pattern = build_groupchat_pattern(agents, shared_context)
    st.markdown("Generating answer...")
    try:
        result, context_variables, last_agent = run_group_chat(
            pattern=pattern,
            messages=(
                "Context from documents: "
                f"{context}\n\n"
                "Conversation history:\n"
                f"{conversation_history}\n\n"
                f"User question: {user_input}"
            ),
            max_rounds=10,
        )
    except Exception as exc:
        error_message = format_llm_error(exc)
        if st.session_state.get("debug"):
            st.session_state.debug_messages.append(("Group chat error", str(exc)))
        return Response(content=error_message)
    formatted_answer = None
    if last_agent == agents.get("refine_agent_final") or last_agent == agents.get(
        "refine_agent_gai"
    ):
        formatted_answer = result.chat_history[-1]["content"]

    if not formatted_answer and shared_context.get("last_answer"):
        formatted_answer = shared_context["last_answer"]

    if not formatted_answer:
        try:
            for item in result.chat_history:
                st.markdown(item)
                if item["name"] in {"class_agent", "imporve_reply_agent"}:
                    formatted_answer = item["content"]
        except Exception:
            formatted_answer = "failed to load chat history"

    if st.session_state.debug:
        st.session_state.debug_messages.append(("Formatted Answer", formatted_answer))
        st.session_state.debug_messages.append(("Feedback", shared_context["feedback"]))

    return Response(content=formatted_answer)
