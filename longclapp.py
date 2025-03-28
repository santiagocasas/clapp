# This script requires Streamlit and LangChain
# Install it with: pip install streamlit openai langchain langchain-openai langchain-community

import streamlit as st
import time
import json
import os
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ChatMessageHistory
from langchain_core.documents import Document

from langchain.callbacks.base import BaseCallbackHandler


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="CLASS Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("images/classAI.png", width=500)

# --- Load API keys and Assistant IDs from file ---
def read_keys_from_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# --- Load system instructions from file ---
def read_prompt_from_file(path):
    with open(path, 'r') as f:
        return f.read()

keys = read_keys_from_file("keys-IDs.json")
Classy_instuctions = read_prompt_from_file("prompts/class_instructions.txt")

# --- Initialize Session State ---
def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "debug" not in st.session_state:
        st.session_state.debug = False
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "memory" not in st.session_state:
        st.session_state.memory = ChatMessageHistory()
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "last_token_count" not in st.session_state:
        st.session_state.last_token_count = 0
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gpt-4o"
    if "greeted" not in st.session_state:
        st.session_state.greeted = False

init_session()

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("🔐 API & Assistants")
    api_key = st.text_input("OpenAI API Key", type="password")

    st.session_state.selected_model = st.selectbox(
        "🧠 Choose LLM model",
        options=["gpt-4o", "gpt-4o-mini", "o3-mini"],
        index=["gpt-4o", "gpt-4o-mini", "o3-mini"].index(st.session_state.selected_model)
    )

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

        # First initialization without streaming
        st.session_state.llm = ChatOpenAI(
                model_name=st.session_state.selected_model,
                openai_api_key=api_key,
                temperature=1.0
        )

        if st.session_state.vector_store is None:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            
            # Get all files from class-data directory
            all_docs = []
            for filename in os.listdir("./class-data"):
                file_path = os.path.join("./class-data", filename)
                
                if filename.endswith('.pdf'):
                    # Handle PDF files
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    all_docs.extend(docs)
                elif filename.endswith(('.txt', '.py')):  # Added .py extension
                    # Handle text and Python files
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        # Create a document with metadata
                        all_docs.append(Document(
                            page_content=text,
                            metadata={"source": filename, "type": "code" if filename.endswith('.py') else "text"}
                        ))

            # Split and process all documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            def sanitize(documents):
                for doc in documents:
                    doc.page_content = doc.page_content.encode("utf-8", "ignore").decode("utf-8")
                return documents
                
            splits = text_splitter.split_documents(all_docs)
            splits = sanitize(splits)
            
            # Create vector store from all documents
            st.session_state.vector_store = FAISS.from_documents(splits, embedding=embeddings)

        # Trigger welcome message once by requesting it from the assistant
        if not st.session_state.greeted:
            greeting = st.session_state.llm.invoke([
                SystemMessage(content=Classy_instuctions),
                HumanMessage(content="Please greet the user and briefly explain what you can do as the CLASS code assistant.")
            ])
            st.session_state.messages.append({"role": "assistant", "content": greeting.content})
            st.session_state.greeted = True

    st.session_state.debug = st.checkbox("🔍 Show Debug Info")
    if st.button("🗑️ Reset Chat"):
        st.session_state.clear()
        st.rerun()

    if st.session_state.last_token_count > 0:
        st.markdown(f"🧮 **Last response token usage:** `{st.session_state.last_token_count}` tokens")

# --- Retrieval + Prompt Construction ---
def build_messages(context, question, system):
    system_msg = SystemMessage(content=system)
    human_msg = HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{question}")
    return [system_msg] + st.session_state.memory.messages + [human_msg]

def retrieve_context(question):
    docs = st.session_state.vector_store.similarity_search(question, k=4)
    return "\n\n".join([doc.page_content for doc in docs])

# --- Chat Input ---
user_input = st.chat_input("Type your prompt here...")

# --- Display Full Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Process New Prompt ---
if user_input:
    # Show user input immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.memory.add_user_message(user_input)
    context = retrieve_context(user_input)
    messages = build_messages(context, user_input, Classy_instuctions)

    # Count prompt tokens using tiktoken if needed
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4")
        st.session_state.last_token_count = len(enc.encode(user_input))
    except:
        st.session_state.last_token_count = 0

    # Stream assistant response
    with st.chat_message("assistant"):
        stream_box = st.empty()
        stream_handler = StreamHandler(stream_box)

        # Second initialization with streaming
        st.session_state.llm = ChatOpenAI(
                model_name=st.session_state.selected_model,
                streaming=True,
                callbacks=[stream_handler],
                openai_api_key=api_key,
                temperature=0.2
        )

        response = st.session_state.llm.invoke(messages)

    st.session_state.memory.add_ai_message(response.content)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response.content})

# --- Debug Info ---
if st.session_state.debug:
    with st.sidebar.expander("🛠️ Context Used"):
        if "context" in locals():
            st.markdown(context)
        else:
            st.markdown("No context retrieved yet.")
    with st.sidebar.expander("📋 System Prompt"):
        st.markdown(Classy_instuctions)
