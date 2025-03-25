import streamlit as st
import openai
import time
import tiktoken

# streamlit configuration
st.set_page_config(
    page_title="CAMELS Agent",
    page_icon='images/logo.png',
    layout="wide",
    initial_sidebar_state="auto",
    menu_items=None
)

# --- Initialize Session State ---
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

init_session_state()

# --- OpenAI Model Selection and API Key ---
api_key = st.text_input("Enter your OpenAI API key:", type="password")
assistant_id = st.text_input("Enter your Assistant ID:", type="default")

if api_key:
    st.session_state["api_key"] = api_key
if assistant_id:
    st.session_state["assistant_id"] = assistant_id

# --- Chat Input ---
prompt = st.chat_input("Type your prompt here...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    # --- Get OpenAI Response ---
    if "api_key" in st.session_state and "assistant_id" in st.session_state:
        openai.api_key = st.session_state["api_key"]
        assistant_id   = st.session_state["assistant_id"]
        
        try:
            # Step 1: Create a thread
            thread = openai.beta.threads.create()

            # Step 2: Add a message to the thread
            message = openai.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=prompt
            )

            # Step 3: Run your assistant
            run = openai.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant_id
            )

            # Wait until the assistant responds (polling)
            while True:
                run_status = openai.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                if run_status.status in ['completed', 'failed', 'cancelled', 'expired']:
                    break
                time.sleep(1)

            # Retrieve assistant's response
            messages = openai.beta.threads.messages.list(thread_id=thread.id)

            assistant_response = ""
            for msg in reversed(messages.data):
                if msg.role == "assistant":
                    assistant_response = msg.content[0].text.value
                    break #Only get the last assistant message

            st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        except Exception as e:
            st.error(f"Error during OpenAI API call: {e}")
    else:
        st.warning("Please enter your OpenAI API key and Assistant ID.")

# --- Display Chat Messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
# Count tokens
if prompt:
    encoding = tiktoken.encoding_for_model("gpt-4")
    token_count = len(encoding.encode(prompt))
    st.sidebar.write(f"Tokens in memory: {token_count}")

