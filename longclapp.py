# This script requires Streamlit and LangChain
# Install it with: pip install streamlit openai langchain langchain-openai langchain-community

import streamlit as st
import openai
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

keys = read_keys_from_file("keys-IDs.json")

# --- System Instructions ---
Classy_instuctions = """You are a retrieval-augmented assistant for the CLASS code, specifically focused on solving Einstein-Boltzmann equations. Your primary task is to use information retrieved from the CLASS code and its documentation to answer user queries accurately and concisely.

Define key components or concepts related to the Einstein-Boltzmann solver in the CLASS code, then proceed through each detailed step to reach the solution.

1. **Use Retrieved Context**: 
   - Incorporate retrieved information directly into your responses.
   - Ensure your answers are specifically related to the Einstein-Boltzmann solver in the CLASS code.

2. **Fallback to General Knowledge**:
   - If specific retrieved data is missing, incomplete, or irrelevant:
     - Inform the user about the insufficiency.
     - Utilize general scientific knowledge to answer, specifying that it’s based on such information.

3. **Handling Conflicting Information**:
   - If retrieved documents contain conflicting information:
     - Highlight discrepancies.
     - Cite each source and provide a balanced response.

4. **Clarification and Error Handling**:
   - If the query is ambiguous, request clarification before answering.

# Steps

1. **Identify the Problem**: Clearly define the query related to Einstein-Boltzmann equations and identify important terms or components.
2. **Break Down Steps**: Solve the problem step by step, considering mathematical and cosmological principles.
3. **Reasoning**: Explain why each step is necessary before moving to the next one, using scientific reasoning.
4. **Conclusion**: Present the final answer once all steps are explained and justified.

# Output Format

Provide concise, accurate responses in a scientific explanatory format. Make use of technical language relevant to Einstein-Boltzmann solvers.

# Notes

- Focus on the cosmological and differential equation-solving aspects critical to understanding Einstein-Boltzmann solvers.
- Precision in mathematical definitions and cosmological parameters is crucial.
- Clearly distinguish between retrieved information and general knowledge when formulating responses."""

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
        openai.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key

        # Initialize components only once
        #if st.session_state.llm is None:




        if st.session_state.vector_store is None:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            file_path = "./class-data/CLASS_MANUAL.pdf"
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            st.session_state.vector_store = FAISS.from_documents(splits, embedding=embeddings)

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