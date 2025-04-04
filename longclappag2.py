# This script requires Streamlit and LangChain
# Install it with: pip install streamlit openai langchain langchain-openai langchain-community

import streamlit as st
import time
import json
import os
import base64
import getpass
from cryptography.fernet import Fernet
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

from pydantic import BaseModel, Field
from typing import Annotated


from autogen import ConversableAgent, LLMConfig

# --- Helper Functions ---
def save_encrypted_key(encrypted_key):
    """Save encrypted key to file"""
    try:
        with open(".encrypted_api_key", "w") as f:
            f.write(encrypted_key)
        return True
    except Exception as e:
        return False

def load_encrypted_key():
    """Load encrypted key from file"""
    try:
        with open(".encrypted_api_key", "r") as f:
            return f.read()
    except FileNotFoundError:
        return None

def read_keys_from_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def read_prompt_from_file(path):
    with open(path, 'r') as f:
        return f.read()
    
class Response:
    def __init__(self, content):
        self.content = content



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
keys = read_keys_from_file("keys-IDs.json")
Classy_instuctions = read_prompt_from_file("prompts/class_instructions.txt")
Rating_instuctions = read_prompt_from_file("prompts/rating_instructions_ag2.txt")

Refine_instuctions = read_prompt_from_file("prompts/class_refinement_ag2.txt")
# --- Initialize Session State ---
def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "debug" not in st.session_state:
        st.session_state.debug = False
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "llmBG" not in st.session_state:
        st.session_state.llmBG = None
    if "memory" not in st.session_state:
        st.session_state.memory = ChatMessageHistory()
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "last_token_count" not in st.session_state:
        st.session_state.last_token_count = 0
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gpt-4o-mini"
    if "greeted" not in st.session_state:
        st.session_state.greeted = False

init_session()

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("🔐 API & Assistants")
    api_key = st.text_input("OpenAI API Key", type="password")
    user_password = st.text_input("Password to encrypt/decrypt API key", type="password")
    
    # When both API key and password are provided
    if api_key and user_password:
        # Create encryption key from password
        key = base64.urlsafe_b64encode(user_password.ljust(32)[:32].encode())
        fernet = Fernet(key)
        
        # If this is a new API key, encrypt and save it
        if "saved_api_key" not in st.session_state or api_key != st.session_state.saved_api_key:
            try:
                # Encrypt the API key
                encrypted_key = fernet.encrypt(api_key.encode())
                
                # Save to session state and file
                st.session_state.saved_api_key = api_key
                st.session_state.encrypted_key = encrypted_key.decode()
                
                # Save to file
                if save_encrypted_key(encrypted_key.decode()):
                    st.success("API key encrypted and saved! ✅")
                else:
                    st.warning("API key encrypted but couldn't save to file! ⚠️")
            except Exception as e:
                st.error(f"Error saving API key: {str(e)}")
    
    # Try to load saved API key if password is provided
    elif user_password and not api_key:
        # Try to load from file first
        encrypted_key = load_encrypted_key()
        if encrypted_key:
            try:
                # Recreate encryption key
                key = base64.urlsafe_b64encode(user_password.ljust(32)[:32].encode())
                fernet = Fernet(key)
                
                # Decrypt the saved key
                decrypted_key = fernet.decrypt(encrypted_key.encode()).decode()
                
                # Set the API key
                api_key = decrypted_key
                st.session_state.saved_api_key = api_key
                st.success("API key loaded successfully! 🔑")
            except Exception as e:
                st.error("Failed to decrypt API key. Wrong password? 🔒")
        else:
            st.warning("No saved API key found. Please enter your API key first. 🔑")

    # Add clear saved key button
    if os.path.exists(".encrypted_api_key"):
        if st.button("🗑️ Clear Saved API Key"):
            try:
                os.remove(".encrypted_api_key")
                if "saved_api_key" in st.session_state:
                    del st.session_state.saved_api_key
                if "encrypted_key" in st.session_state:
                    del st.session_state.encrypted_key
                st.success("Saved API key cleared! 🗑️")
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing saved key: {str(e)}")

    st.session_state.selected_model = st.selectbox(
        "🧠 Choose LLM model",
        options=["gpt-4o-mini", "gpt-4o", "o3-mini"],
        index=["gpt-4o-mini", "gpt-4o", "o3-mini"].index(st.session_state.selected_model)
    )

    st.write("### Response Mode")
    col1, col2 = st.columns([1, 2])
    with col1:
        mode_is_fast = st.toggle("Fast Mode", value=True)
    with col2:
        if mode_is_fast:
            st.caption("✨ Quick responses with good quality (recommended for most uses)")
        else:
            st.caption("🎯 Detailed, more refined responses (may take longer)")
    
    st.markdown("---")  # Add a separator for better visual organization

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        
        # First initialization without streaming
        st.session_state.llm = ChatOpenAI(
                model_name=st.session_state.selected_model,
                openai_api_key=api_key,
                temperature=1.0
        )
        st.session_state.llmBG = ChatOpenAI(
                model_name=st.session_state.selected_model,
                openai_api_key=api_key,
                temperature=0.2
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
                elif filename.endswith(('.txt', '.py', '.ini')):  # Added .py extension
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

def build_messages_rating(context, question, answer, system):
    system_msg = SystemMessage(content=system)
    human_msg = HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{question}\n\nAI Answer:\n{answer}")
    return [system_msg] + st.session_state.memory.messages + [human_msg]

def build_messages_refinement(context, question, answer, feedback, system):
    system_msg = SystemMessage(content=system)
    human_msg = HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{question}\n\nAI Answer:\n{answer}\n\nReviewer Feedback:\n{feedback}")
    return [system_msg] + st.session_state.memory.messages + [human_msg]

def format_memory_messages(memory_messages):
    formatted = ""
    for msg in memory_messages:
        role = msg.type.capitalize()  # 'human' -> 'Human'
        content = msg.content
        formatted += f"{role}: {content}\n\n"
    return formatted.strip()


def retrieve_context(question):
    docs = st.session_state.vector_store.similarity_search(question, k=4)
    return "\n\n".join([doc.page_content for doc in docs])

def call_ai(context, user_input):
    if mode_is_fast:
        messages = build_messages(context, user_input, Classy_instuctions)

        response = st.session_state.llm.invoke(messages)
    else:

        # GP2 workflow
        st.markdown("Thinking ... ")
        

        # Curriculum designer
        generator_message = Classy_instuctions

        # Lesson planner
        rating_message = Rating_instuctions

        # Formatter
        reviewer_message = Refine_instuctions


        class Feedback(BaseModel):
            grade: Annotated[int, Field(description="Score from 1 to 10")]
            improvement_instructions: Annotated[str, Field(description="Advice on how to improve")]


        llm_config = LLMConfig(api_type="openai", model=st.session_state.selected_model)

        llm_config2 = LLMConfig(api_type="openai", model=st.session_state.selected_model,response_format=Feedback)



        with llm_config:
            classy_agent = ConversableAgent(
                name="classy_agent",
                system_message=generator_message,
            )

            review_agent = ConversableAgent(
                name="classy_agent",
                system_message=reviewer_message,
            )


        with llm_config2:
            rating_agent = ConversableAgent(
                name="rating_agent",
                system_message=rating_message,
            )

        memory_str = format_memory_messages(st.session_state.memory.messages)

         # 2. Construct full message with memory + context

        

        # Step 1: Run first two agents
        chat_results = classy_agent.initiate_chats(
            [
                {
                    "recipient": classy_agent,
                    "message": f"""
                        Here is the conversation so far:

                        {memory_str}

                        Here is some relevant context:

                        {context}

                        Now answer this question:

                        {user_input}
                        """,
                    "max_turns": 1,
                    "summary_method": "last_msg",
                },
                {
                    "recipient": rating_agent,
                    "message": f"""
                        Here is the conversation so far:

                        {memory_str}

                        Here is some relevant context:

                        {context}

                        Here is the original user question:

                        {user_input}

                        Now review the previous ai reply to the user question""",
                    "max_turns": 1,
                    "summary_method": "last_msg",
                },
            ]
        )

        # Step 2: Parse the rating agent's summary into the Feedback model
        rating_summary = chat_results[1].summary

        try:
            feedback_data = json.loads(rating_summary)
            feedback = Feedback(**feedback_data)
        except Exception as e:
            st.error(f"Failed to parse feedback: {e}")
            feedback = Feedback(grade=0, improvement_instructions="(Invalid feedback)")

        # Show summaries
        #st.text(f"\n\nClassy Agent Summary:\n {chat_results[0].summary}")
        #st.text(f"\n\nRating Agent Summary:\n {feedback}")
        #st.text(f"\n\nRating Given: {feedback.grade}")

        # Step 3: Conditionally run the review agent
        if feedback.grade <= 8:
            review_result = classy_agent.initiate_chat(
                recipient=review_agent,
                message=f"""
                    Here is the conversation so far:

                    {memory_str}

                    Here is some relevant context:

                    {context}

                    Here is the original user question:

                    {user_input}

                    Here is the Ai answer to be improved:

                    {chat_results[0].summary}

                    Here is the provided feedback:

                    {feedback.improvement_instructions}

                    Now update the Ai answer based on the provided feedback""",
                max_turns=1,
                summary_method="last_msg"
            )
            #st.text(f"\n\nReview Agent Summary:\n {review_result.summary}")

            response = Response(content=review_result.summary)
        else:
            #st.text("\n\nReview not needed. Rating is sufficient.")

            response = Response(content=chat_results[0].summary)
        
        # Then use it like:



        #messages = build_messages(context, user_input, Classy_instuctions)

        #response_1 = st.session_state.llmBG.invoke(messages)
        
        # Send grading prompt
        #messages = build_messages_rating(context, user_input,response_1.content, Rating_instuctions)

        #grading_response = st.session_state.llmBG.invoke(messages)

        #try:
        #    grading_data = json.loads(grading_response.content)
        #    st.markdown(f"**Rating:** {grading_data['rating']}/10")
        #    st.markdown(f"**Feedback:** {grading_data['feedback']}")

        #    if grading_data['rating'] <= 8:
                #st.warning("⚠️ Rating below threshold — attempting to improve the answer.")

        #        messages = build_messages_refinement(context, user_input,response_1.content,grading_data['feedback'], Refine_instuctions)
        #        response = st.session_state.llmBG.invoke(messages)


            #else: 
        #        response = response_1
        #except json.JSONDecodeError:
        #    st.warning("⚠️ Couldn't parse grading response as JSON.")
        #    st.text(grading_response.content)

        #    response = response_1
        # rate this response by ai

 
    return response


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

        response = call_ai(context,user_input)

    st.session_state.memory.add_ai_message(response.content)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response.content})

    if not mode_is_fast:
        st.markdown(response.content)

# --- Debug Info ---
if st.session_state.debug:
    with st.sidebar.expander("🛠️ Context Used"):
        if "context" in locals():
            st.markdown(context)
        else:
            st.markdown("No context retrieved yet.")
    with st.sidebar.expander("📋 System Prompt"):
        st.markdown(Classy_instuctions)
