from typing import Annotated

import streamlit as st
from autogen import ConversableAgent, LLMConfig, UpdateSystemMessage
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group import (
    AgentNameTarget,
    AgentTarget,
    ContextVariables,
    OnCondition,
    ReplyResult,
    StringLLMCondition,
    TerminateTarget,
)
from autogen.agentchat.group.patterns import AutoPattern

from clapp.config import (
    BLABLADOR_MODELS,
    GEMINI_MODELS,
    GPT_MODELS,
    get_openai_base_url,
    normalize_base_url,
)
from clapp.prompts import load_prompts

PROMPTS = load_prompts()


def review_reply(
    feedback: Annotated[
        str,
        "Feedback on improving this reply to be accurate and relavant for the user prompt",
    ],
    rating: Annotated[int, "The rating of the reply on a scale of 1 to 10"],
    context_variables: ContextVariables,
) -> ReplyResult:
    context_variables["feedback"] = feedback
    context_variables["rating"] = rating
    context_variables["revisions"] += 1

    messages = list(st.session_state.agents["class_agent"].chat_messages.values())[0]
    reply = None
    for item in messages:
        if item["name"] in {"class_agent", "improve_reply_agent"}:
            reply = item["content"]

    if reply:
        context_variables["last_answer"] = reply

    if rating < 8 and context_variables["revisions"] < 3:
        return ReplyResult(
            context_variables=context_variables,
            target=AgentNameTarget("improve_reply_agent"),
            message=f"Please revise the answer considering this feedback {feedback}",
        )
    if rating >= 8:
        return ReplyResult(
            context_variables=context_variables,
            target=AgentNameTarget("improve_reply_agent_final"),
            message="The answer is already of sufficient quality. Focus on formatting the reply",
        )
    return ReplyResult(
        context_variables=context_variables,
        target=AgentNameTarget("improve_reply_agent_final"),
        message=f"Please revise the answer considering this feedback {feedback}",
    )


def get_agents():
    selected_model = st.session_state.selected_model
    if selected_model in GPT_MODELS:
        openai_base_url = get_openai_base_url()
        initial_config = LLMConfig(
            api_type="openai",
            model=selected_model,
            temperature=0.2,
            api_key=st.session_state.get("saved_api_key"),
            base_url=openai_base_url,
        )
        review_config = LLMConfig(
            api_type="openai",
            model=selected_model,
            temperature=0.5,
            api_key=st.session_state.get("saved_api_key"),
            base_url=openai_base_url,
        )
        class_agent = ConversableAgent(
            name="class_agent",
            system_message=PROMPTS["initial"],
            description="Initial agent that answers user prompt. Expert in the CLASS code",
            human_input_mode="NEVER",
            llm_config=initial_config,
        )
        review_agent = ConversableAgent(
            name="review_agent",
            update_agent_state_before_reply=[UpdateSystemMessage(PROMPTS["review"])],
            human_input_mode="NEVER",
            description="Reviews the AI answer to user prompt",
            llm_config=review_config,
            functions=review_reply,
        )
        refine_agent = ConversableAgent(
            name="improve_reply_agent",
            update_agent_state_before_reply=[UpdateSystemMessage(PROMPTS["refine"])],
            human_input_mode="NEVER",
            description="Improves the AI reply by taking into account the feedback",
            llm_config=initial_config,
        )
        refine_agent_final = ConversableAgent(
            name="improve_reply_agent_final",
            update_agent_state_before_reply=[UpdateSystemMessage(PROMPTS["refine"])],
            human_input_mode="NEVER",
            description="Improves the AI reply by taking into account the feedback",
            llm_config=initial_config,
        )
        class_agent.handoffs.set_after_work(AgentTarget(review_agent))
        review_agent.handoffs.set_after_work(AgentTarget(refine_agent))
        refine_agent.handoffs.set_after_work(AgentTarget(review_agent))
        refine_agent_final.handoffs.set_after_work(TerminateTarget())
        refine_agent.handoffs.add_llm_conditions(
            [
                OnCondition(
                    target=AgentTarget(refine_agent_final),
                    condition=StringLLMCondition(
                        prompt=(
                            "The reply to the latest user question has been reviewd and received "
                            "a favarable rating (equivalent to 7 or higher)"
                        )
                    ),
                )
            ]
        )
        st.session_state.agents = {
            "class_agent": class_agent,
            "review_agent": review_agent,
            "refine_agent": refine_agent,
            "refine_agent_final": refine_agent_final,
            "initial_config": initial_config,
            "refine_agent_gai": None,
        }
        return st.session_state.agents
    blablador_models = st.session_state.get("blablador_models") or BLABLADOR_MODELS
    if selected_model in GEMINI_MODELS:
        initial_config_gai = LLMConfig(
            api_type="google",
            model=selected_model,
            temperature=0.2,
            api_key=st.session_state.get("saved_api_key_gai"),
        )
        review_config_gai = LLMConfig(
            api_type="google",
            model=selected_model,
            temperature=0.5,
            api_key=st.session_state.get("saved_api_key_gai"),
        )
        class_agent_gai = ConversableAgent(
            name="class_agent",
            system_message=PROMPTS["initial"],
            description="Initial agent that answers user prompt. Expert in the CLASS code",
            human_input_mode="NEVER",
            llm_config=initial_config_gai,
        )
        refine_agent_gai = ConversableAgent(
            name="improve_reply_agent",
            update_agent_state_before_reply=[UpdateSystemMessage(PROMPTS["refine"])],
            human_input_mode="NEVER",
            description="Improves the AI reply by taking into account the feedback",
            llm_config=initial_config_gai,
        )
        review_agent_gai = ConversableAgent(
            name="review_agent",
            update_agent_state_before_reply=[UpdateSystemMessage(PROMPTS["review"])],
            human_input_mode="NEVER",
            description="Reviews the AI answer to user prompt",
            llm_config=review_config_gai,
        )
        class_agent_gai.handoffs.set_after_work(AgentTarget(review_agent_gai))
        review_agent_gai.handoffs.set_after_work(AgentTarget(refine_agent_gai))
        refine_agent_gai.handoffs.set_after_work(TerminateTarget())
        st.session_state.agents = {
            "class_agent_gai": class_agent_gai,
            "review_agent_gai": review_agent_gai,
            "refine_agent_gai": refine_agent_gai,
            "initial_config_gai": initial_config_gai,
            "refine_agent_final": None,
        }
        return st.session_state.agents
    if st.session_state.get("saved_api_key_blablador") and (
        selected_model in blablador_models
        or selected_model not in GEMINI_MODELS + GPT_MODELS
    ):
        blablador_base_url = normalize_base_url(st.session_state.blablador_base_url)
        initial_config = LLMConfig(
            api_type="openai",
            model=selected_model,
            temperature=0.2,
            api_key=st.session_state.get("saved_api_key_blablador"),
            base_url=blablador_base_url,
        )
        review_config = LLMConfig(
            api_type="openai",
            model=selected_model,
            temperature=0.5,
            api_key=st.session_state.get("saved_api_key_blablador"),
            base_url=blablador_base_url,
        )
        class_agent = ConversableAgent(
            name="class_agent",
            system_message=PROMPTS["initial"],
            description="Initial agent that answers user prompt. Expert in the CLASS code",
            human_input_mode="NEVER",
            llm_config=initial_config,
        )
        review_agent = ConversableAgent(
            name="review_agent",
            update_agent_state_before_reply=[UpdateSystemMessage(PROMPTS["review"])],
            human_input_mode="NEVER",
            description="Reviews the AI answer to user prompt",
            llm_config=review_config,
            functions=review_reply,
        )
        refine_agent = ConversableAgent(
            name="improve_reply_agent",
            update_agent_state_before_reply=[UpdateSystemMessage(PROMPTS["refine"])],
            human_input_mode="NEVER",
            description="Improves the AI reply by taking into account the feedback",
            llm_config=initial_config,
        )
        refine_agent_final = ConversableAgent(
            name="improve_reply_agent_final",
            update_agent_state_before_reply=[UpdateSystemMessage(PROMPTS["refine"])],
            human_input_mode="NEVER",
            description="Improves the AI reply by taking into account the feedback",
            llm_config=initial_config,
        )
        class_agent.handoffs.set_after_work(AgentTarget(review_agent))
        review_agent.handoffs.set_after_work(AgentTarget(refine_agent))
        refine_agent.handoffs.set_after_work(AgentTarget(review_agent))
        refine_agent_final.handoffs.set_after_work(TerminateTarget())
        refine_agent.handoffs.add_llm_conditions(
            [
                OnCondition(
                    target=AgentTarget(refine_agent_final),
                    condition=StringLLMCondition(
                        prompt=(
                            "The reply to the latest user question has been reviewd and received "
                            "a favarable rating (equivalent to 7 or higher)"
                        )
                    ),
                )
            ]
        )
        st.session_state.agents = {
            "class_agent": class_agent,
            "review_agent": review_agent,
            "refine_agent": refine_agent,
            "refine_agent_final": refine_agent_final,
            "initial_config": initial_config,
            "refine_agent_gai": None,
        }
        return st.session_state.agents
    return {}


def run_group_chat(pattern, messages, max_rounds):
    return initiate_group_chat(
        pattern=pattern, messages=messages, max_rounds=max_rounds
    )


def build_groupchat_pattern(agents, shared_context_data):
    selected_model = st.session_state.selected_model
    shared_context = ContextVariables(data=shared_context_data)
    if selected_model in GEMINI_MODELS:
        return AutoPattern(
            initial_agent=agents["class_agent_gai"],
            agents=[
                agents["class_agent_gai"],
                agents["review_agent_gai"],
                agents["refine_agent_gai"],
            ],
            group_manager_args={"llm_config": agents["initial_config_gai"]},
            context_variables=shared_context,
        )
    return AutoPattern(
        initial_agent=agents["class_agent"],
        agents=[
            agents["class_agent"],
            agents["review_agent"],
            agents["refine_agent"],
            agents["refine_agent_final"],
        ],
        group_manager_args={"llm_config": agents["initial_config"]},
        context_variables=shared_context,
    )
