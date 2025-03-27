# This script requires Streamlit.
# Install it with: pip install streamlit openai

import streamlit as st
import openai
import time
import json

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="CLASS Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items=None
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("images/classAI.png", width=500)

# --- Load API keys and Assistant IDs from file ---
def read_keys_from_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

keys = read_keys_from_file("keys-IDs.json")

# --- Initialize Session State ---
def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None
    if "debug" not in st.session_state:
        st.session_state.debug = False
    if "raw_response" not in st.session_state:
        st.session_state.raw_response = ""
    if "raw_eval" not in st.session_state:
        st.session_state.raw_eval = ""

init_session()

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("🔐 API & Assistants")
    api_key = st.text_input("OpenAI API Key", type="password")
    assistant_ids = {
        "CLASS Assistant": st.text_input("CLASS Assistant", value=keys["assistant_ID"]),
        "Evaluator": st.text_input("Evaluator", value=keys["evaluator_ID"]),
        "Optimizer": st.text_input("Optimizer", value=keys["optimizer_ID"])
    }
    st.markdown("---")
    st.session_state.debug = st.checkbox("🔍 Show Debug Info")
    if st.button("🗑️ Reset Chat"):
        st.session_state.clear()
        st.rerun()

# --- Set OpenAI key ---
if api_key:
    openai.api_key = api_key

# --- Chat Input ---
user_input = st.chat_input("Type your prompt here...")

# --- Display Full Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Process New Prompt ---
if user_input:
    # Display user input immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Create thread if it doesn't exist
    if not st.session_state.thread_id:
        thread = openai.beta.threads.create()
        st.session_state.thread_id = thread.id

    thread_id = st.session_state.thread_id

    # Send prompt to assistant
    openai.beta.threads.messages.create(thread_id=thread_id, role="user", content=user_input)
    run = openai.beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_ids["CLASS Assistant"])
    while True:
        run_status = openai.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        if run_status.status in ["completed", "failed", "cancelled", "expired"]:
            break
        time.sleep(1)
    response_msgs = openai.beta.threads.messages.list(thread_id=thread_id, run_id=run.id)
    st.session_state.raw_response = next((m.content[0].text.value for m in reversed(response_msgs.data) if m.role == "assistant"), "")

    # Evaluate the response
    eval_prompt = f"""
    ## User Prompt
    {user_input}

    ## Assistant Response
    {st.session_state.raw_response}

    Please evaluate the assistant's response. Return JSON with keys: score, verdict, feedback.
    """
    eval_thread = openai.beta.threads.create()
    openai.beta.threads.messages.create(thread_id=eval_thread.id, role="user", content=eval_prompt)
    run_eval = openai.beta.threads.runs.create(thread_id=eval_thread.id, assistant_id=assistant_ids["Evaluator"])
    while True:
        status = openai.beta.threads.runs.retrieve(thread_id=eval_thread.id, run_id=run_eval.id)
        if status.status in ["completed", "failed", "cancelled", "expired"]:
            break
        time.sleep(1)
    eval_msgs = openai.beta.threads.messages.list(thread_id=eval_thread.id, run_id=run_eval.id)
    st.session_state.raw_eval = next((m.content[0].text.value for m in reversed(eval_msgs.data) if m.role == "assistant"), "")
    try:
        evaluation = json.loads(st.session_state.raw_eval)
    except json.JSONDecodeError:
        evaluation = {"score": None, "verdict": "Parse Error", "feedback": st.session_state.raw_eval}

    # Optimize final response
    opt_prompt = f"""
    ## User Prompt
    {user_input}

    ## Assistant Response
    {st.session_state.raw_response}

    ## Evaluation Feedback
    {evaluation}

    Please provide an optimized version of the assistant's answer based on the feedback.
    """
    opt_thread = openai.beta.threads.create()
    openai.beta.threads.messages.create(thread_id=opt_thread.id, role="user", content=opt_prompt)
    run_opt = openai.beta.threads.runs.create(thread_id=opt_thread.id, assistant_id=assistant_ids["Optimizer"])
    while True:
        status = openai.beta.threads.runs.retrieve(thread_id=opt_thread.id, run_id=run_opt.id)
        if status.status in ["completed", "failed", "cancelled", "expired"]:
            break
        time.sleep(1)
    opt_msgs = openai.beta.threads.messages.list(thread_id=opt_thread.id, run_id=run_opt.id)
    optimized = next((m.content[0].text.value for m in reversed(opt_msgs.data) if m.role == "assistant"), "")

    # Stream optimized answer
    with st.chat_message("assistant"):
        stream_box = st.empty()
        typed = ""
        for char in optimized:
            typed += char
            stream_box.markdown(typed + "▌")
            time.sleep(0.01)
        stream_box.markdown(typed)

    # Save both user and assistant messages
    st.session_state.messages.extend([
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": optimized}
    ])

# --- Debug Panel ---
if st.session_state.debug:
    with st.sidebar.expander("🛠️ CLASS Assistant Raw Output"):
        st.markdown(st.session_state.raw_response or "No response yet.")
    with st.sidebar.expander("🧪 Evaluation Feedback"):
        st.markdown(st.session_state.raw_eval or "No feedback yet.")