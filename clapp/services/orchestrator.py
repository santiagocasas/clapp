import streamlit as st

from clapp.agents.groupchat import build_groupchat_pattern, get_agents, run_group_chat
from clapp.config import BLABLADOR_MODELS
from clapp.execution.code_execution import call_code as run_code
from clapp.rag.pipeline import build_messages, format_memory_messages


class Response:
    def __init__(self, content):
        self.content = content


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
    if st.session_state.selected_model in BLABLADOR_MODELS:
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
        for chunk in st.session_state.llm.stream(messages):
            response.append(chunk.content)

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
