import streamlit as st
import openai
import time
import tiktoken

# streamlit configuration
st.set_page_config(
    page_title="CLASS Agent",
    page_icon='images/classAI.png',
    layout="wide",
    initial_sidebar_state="auto",
    menu_items=None
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("images/classAI.png", width=500)

# --- Initialize Session State ---
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None

init_session_state()

# --- Sidebar: OpenAI API Key and Assistant ID ---
with st.sidebar:
    st.header("🔑 API Settings")

    api_key = st.text_input("OpenAI API Key", type="password")

    assistant_options = {
        "Default (My CLASS Assistant)": "asst_pq9CSPIkCoLvx3XkLdBUZbGV",
        "Custom...": "custom"
    }

    selected_option = st.selectbox("Select Assistant:", list(assistant_options.keys()))
    
    if assistant_options[selected_option] == "custom":
        assistant_id = st.text_input("Enter your custom Assistant ID:")
    else:
        assistant_id = assistant_options[selected_option]

    if st.button("🗑️ Reset Chat"):
        for key in ["messages", "thread_id"]:
            st.session_state.pop(key, None)
        st.rerun()

# Store credentials
if api_key:
    st.session_state["api_key"] = api_key
if assistant_id:
    st.session_state["assistant_id"] = assistant_id

# --- Chat Input ---
prompt = st.chat_input("Type your prompt here...")

# --- Token Counting ---
if prompt:
    encoding = tiktoken.encoding_for_model("gpt-4")
    token_count = len(encoding.encode(prompt))
    with st.sidebar:
        st.markdown(f"🧮 **Tokens in memory:** `{token_count}`")

# --- Handle Chat and API Call ---
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    if "api_key" in st.session_state and "assistant_id" in st.session_state:
        openai.api_key = st.session_state["api_key"]
        assistant_id = st.session_state["assistant_id"]

        try:
            # Step 1: Create thread if it doesn't exist
            if not st.session_state.get("thread_id"):
                thread = openai.beta.threads.create()
                st.session_state["thread_id"] = thread.id

            thread_id = st.session_state["thread_id"]

            # Step 2: Add message to thread
            openai.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=prompt
            )

            # Step 3: Run assistant
            run = openai.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id
            )

            # Step 4: Wait until the run is done
            while True:
                run_status = openai.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run.id
                )
                if run_status.status in ['completed', 'failed', 'cancelled', 'expired']:
                    break
                time.sleep(1)

            # Step 5: Get ONLY messages generated in this run
            response_messages = openai.beta.threads.messages.list(
                thread_id=thread_id,
                run_id=run.id
            )

            assistant_response = None
            for msg in reversed(response_messages.data):
                if msg.role == "assistant":
                    assistant_response = msg.content[0].text.value
                    break

            # Step 6: Save the assistant response
            if assistant_response:
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        except:
            raise ValueError("A problem appeared") 
    else:
        st.warning("Please enter your OpenAI API key and Assistant ID.")

# --- Display Chat Messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
