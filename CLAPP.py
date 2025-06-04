# This script requires Streamlit and LangChain
# Install it with: pip install streamlit openai langchain langchain-openai langchain-community

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"  # Optional: use a writable cache

import streamlit as st
import time
import json
import os
import base64
import getpass
from collections import defaultdict


from cryptography.fernet import Fernet
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from autogen.agentchat.group import OnCondition, StringLLMCondition

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI

from pydantic import BaseModel, Field
from typing import Annotated


from autogen import ConversableAgent, LLMConfig, UpdateSystemMessage, ContextExpression

from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import AutoPattern
from autogen.agentchat.group import ReplyResult, AgentNameTarget, OnContextCondition, ExpressionContextCondition
from autogen.agentchat.group import AgentTarget, RevertToUserTarget, TerminateTarget, NestedChatTarget
from typing import Annotated
from autogen.agentchat.group import ContextVariables

#import google.generativeai as genai

import tempfile
from autogen.coding import LocalCommandLineCodeExecutor, CodeBlock
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
import io
from PIL import Image
import re
import subprocess
import sys
from typing import Tuple
import contextlib  # for contextlib.contextmanager

def get_all_docs_from_class_data():
    all_docs = []
    for filename in os.listdir("./class-data"):
        file_path = os.path.join("./class-data", filename)
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)
        elif filename.endswith(('.txt', '.py', '.ini')):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                all_docs.append(Document(
                    page_content=text,
                    metadata={"source": filename, "type": "code" if filename.endswith('.py') else "text"}
                ))
    return all_docs
# --- Helper Functions ---
def save_encrypted_key(encrypted_key, username):
    """Save encrypted key to file with username prefix"""
    if not username:
        username = 'anon'
    try:
        filename = f"{username}_encrypted_api_key" if username else ".encrypted_api_key"
        with open(filename, "w") as f:
            f.write(encrypted_key)
        return True
    except Exception as e:
        return False

def load_encrypted_key(username):
    """Load encrypted key from file with username prefix"""
    if not username:
        username = 'anon'
    try:
        filename = f"{username}_encrypted_api_key" if username else ".encrypted_api_key"
        with open(filename, "r") as f:
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
    page_title="CLAPP Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto"
)

st.markdown("# CLAPP: CLASS LLM Agent for Pair Programming")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("images/CLAPP.png", width=400)

# --- Model Lists ---
GPT_MODELS = [ "gpt-4o-mini", "gpt-4o", "gpt-4.1"]
GEMINI_MODELS = [ "gemini-2.0-flash", "gemini-1.5-flash","gemini-2.5-flash-preview-05-20"]
#ALL_MODELS = GPT_MODELS + GEMINI_MODELS

# New prompts for the swarm
Initial_Agent_Instructions = read_prompt_from_file("prompts/class_instructions.txt") # Reuse or adapt class_instructions
Refine_Agent_Instructions = read_prompt_from_file("prompts/class_refinement.txt") # Instructions on imporving an answer
Review_Agent_Instructions = read_prompt_from_file("prompts/review_instructions.txt") # Adapt rating_instructions
Formatting_Agent_Instructions = read_prompt_from_file("prompts/formatting_instructions.txt") # New prompt file
Code_Execution_Agent_Instructions = read_prompt_from_file("prompts/codeexecutor_instructions.txt") # New prompt file

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
        st.session_state.selected_model = None
    if "greeted" not in st.session_state:
        st.session_state.greeted = False
    if "debug_messages" not in st.session_state:
        st.session_state.debug_messages = []
    if "saved_api_key" not in st.session_state:
        st.session_state.saved_api_key = None
    if "saved_api_key_gai" not in st.session_state:
        st.session_state.saved_api_key_gai = None


init_session()



# --- Sidebar Configuration ---
with st.sidebar:
    st.header("🔐 API & Assistants")
    api_key = st.text_input("1. OpenAI API Key", type="password")
    api_key_gai = st.text_input("1. Gemini API Key", type="password")

    ## --- Load API Key (direct, no encryption) ---
    #if st.button("🔓 Load API Key(s) into session"):
    if api_key:
        st.session_state.saved_api_key = api_key
    if api_key_gai:
        st.session_state.saved_api_key_gai = api_key_gai
    #    st.success("API key(s) loaded into session.")
    #    st.rerun()

    username = st.text_input("2. Username (for loading or saving API key)", placeholder="Enter your username")
    user_password = st.text_input("3. Password to encrypt/decrypt API key", type="password")

    # File existence checks
    username_display = username if username else 'anon'
    openai_file = f"{username_display}_encrypted_api_key"
    gemini_file = f"{username_display}_gai_encrypted_api_key"
    openai_file_exists = os.path.exists(openai_file)
    gemini_file_exists = os.path.exists(gemini_file)

    # Session state checks
    openai_loaded = bool(st.session_state.get("saved_api_key"))
    gemini_loaded = bool(st.session_state.get("saved_api_key_gai"))

    # Status display
    st.markdown(f"OpenAI Key: {'✅ Ready' if openai_loaded else '❌ No Keys'} | Saved: {'🗄️' if openai_file_exists else '—'}")
    st.markdown(f"Gemini Key: {'✅ Ready' if gemini_loaded else '❌ No Keys'} | Saved: {'🗄️' if gemini_file_exists else '—'}")

    # --- Save API Key as encrypted file ---
    if (openai_loaded or gemini_loaded) and user_password and username and (not openai_file_exists or not gemini_file_exists):
        if st.button("💾 Save API Key(s) as encrypted file"):
            key = base64.urlsafe_b64encode(user_password.ljust(32)[:32].encode())
            fernet = Fernet(key)
            try:
                if openai_loaded and not openai_file_exists:
                    encrypted_key = fernet.encrypt(st.session_state.saved_api_key.encode())
                    if save_encrypted_key(encrypted_key.decode(), username_display):
                        st.success("OpenAI API key encrypted and saved! ✅")
                    else:
                        st.warning("OpenAI API key encrypted but couldn't save to file! ⚠️")
                if gemini_loaded and not gemini_file_exists:
                    encrypted_key_gai = fernet.encrypt(st.session_state.saved_api_key_gai.encode())
                    if save_encrypted_key(encrypted_key_gai.decode(), username_display+'_gai'):
                        st.success("Gemini API key encrypted and saved! ✅")
                    else:
                        st.warning("Gemini API key encrypted but couldn't save to file! ⚠️")
            except Exception as e:
                st.error(f"Error saving API key: {str(e)}")
            st.rerun()

    # --- Load Saved API Key (from encrypted file) ---
    if st.button("🔐 Load Saved API Key(s)"):
        if not username or not user_password:
            st.error("Please enter both username and password to load saved API key(s).")
        else:
            key = base64.urlsafe_b64encode(user_password.ljust(32)[:32].encode())
            fernet = Fernet(key)
            error = False
            try:
                if openai_file_exists:
                    encrypted_key = load_encrypted_key(username_display)
                    decrypted_key = fernet.decrypt(encrypted_key.encode()).decode()
                    st.session_state.saved_api_key = decrypted_key
                    
                    st.success("OpenAI API key loaded from encrypted file! 🔑")
                if gemini_file_exists:
                    encrypted_key_gai = load_encrypted_key(username_display+'_gai')
                    decrypted_key_gai = fernet.decrypt(encrypted_key_gai.encode()).decode()
                    st.session_state.saved_api_key_gai = decrypted_key_gai
   
                    st.success("Gemini API key loaded from encrypted file! 🔑")
            except Exception as e:
                st.error("Failed to decrypt API key(s): Please check your username and password.")
                error = True
            if not error:
                # Set llm_initialized if a key is loaded and a model is selected
                if (
                    (st.session_state.saved_api_key and st.session_state.selected_model in GPT_MODELS) or
                    (st.session_state.saved_api_key_gai and st.session_state.selected_model in GEMINI_MODELS)
                ):
                    st.session_state.llm_initialized = True
                st.rerun()

    # --- Clear Saved API Key (delete encrypted file and clear session) ---
    if (openai_file_exists or gemini_file_exists):
        if st.button("🗑️ Clear Saved API Key(s)"):
            deleted_files = False
            error_message = ""
            try:
                if openai_file_exists:
                    os.remove(openai_file)
                    deleted_files = True
                if gemini_file_exists:
                    os.remove(gemini_file)
                    deleted_files = True
            except Exception as e:
                error_message += f"Error clearing file: {str(e)}\n"
            for k in ["saved_api_key", "saved_api_key_gai", "encrypted_key", "encrypted_key_gai"]:
                if k in st.session_state:
                    del st.session_state[k]
            if deleted_files:
                st.info("Saved API key(s) cleared. Reloading page...")
                time.sleep(1)
                st.rerun()
            elif error_message:
                st.error(error_message)
            else:
                st.warning("No saved API keys found to delete.")
    
    
                     
    if st.session_state.saved_api_key:
        api_key = st.session_state.saved_api_key
    if st.session_state.saved_api_key_gai:
        api_key_gai = st.session_state.saved_api_key_gai
    

    


    OPTIONS = []
    if api_key_gai:
        OPTIONS += GEMINI_MODELS
    if api_key:
        OPTIONS += GPT_MODELS

    if OPTIONS:
        st.markdown("---")  # Add a separator for better visual organization
    
        st.session_state.selected_model = OPTIONS[0]
        st.session_state.selected_model = st.selectbox(
            "4. Choose LLM model",
            options=OPTIONS,
            index=OPTIONS.index(st.session_state.selected_model)
        )
    else:
        st.session_state.selected_model = None


    # Check if model has changed
    if "previous_model" not in st.session_state:
        st.session_state.previous_model = st.session_state.selected_model
    elif st.session_state.previous_model != st.session_state.selected_model:
        # Reset relevant state variables when model changes
        #st.session_state.vector_store = None
        st.session_state.greeted = False
        st.session_state.messages = []
        st.session_state.memory = ChatMessageHistory()
        st.session_state.previous_model = st.session_state.selected_model
        st.info("Model changed! Chat has been reset.")
    if st.session_state.selected_model in GEMINI_MODELS:
        if api_key_gai:
            st.session_state.llm_initialized = True        
    elif st.session_state.selected_model in GPT_MODELS and api_key:
        st.session_state.llm_initialized = True
        
    if OPTIONS:
        st.write("### Response Mode")
        mode = st.radio(
            "",
            options=["Fast Mode", "Deep Thought Mode"],
            index=0 if st.session_state.get("mode_is_fast", "Fast Mode") == "Fast Mode" else 1,
            horizontal=True,
            key="mode_is_fast"
        )

        st.markdown("<div style='height: 0.5em'></div>", unsafe_allow_html=True)
        desc_cols = st.columns(2)
        with desc_cols[0]:
            st.caption("✨ **Fast Mode**: Single agent setup, quick responses with good quality.")
        with desc_cols[1]:
            st.caption("🎯 **Deep Thought Mode**: Multi-agent setup, more refined responses, takes longer.")
        
    else:
        st.session_state.mode_is_fast = "Fast Mode"


    st.markdown("---")  # Add a separator for better visual organization

    # --- Helper for RAG Embedding Generation ---
    def generate_and_save_embedding(index_path):
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        all_docs = get_all_docs_from_class_data()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        def sanitize(documents):
            for doc in documents:
                doc.page_content = doc.page_content.encode("utf-8", "ignore").decode("utf-8")
            return documents
        splits = text_splitter.split_documents(all_docs)
        splits = sanitize(splits)
        st.session_state.vector_store = FAISS.from_documents(splits, embedding=embeddings)
        st.session_state.vector_store.save_local(index_path)
    
    # --- RAG/Embedding Section ---
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    embedding_status = st.empty()
    index_path = "my_faiss_index"
    index_file = os.path.join(index_path, "index.faiss")
    index_exists = os.path.exists(index_file)

    if st.session_state.vector_store:
        st.markdown("✅ Embedding loaded from file")
        if st.button("🔄 Regenerate embedding"):
            embedding_status.info("🔄 Processing and embedding your RAG data... This might take a moment! ⏳")
            generate_and_save_embedding(index_path)
            embedding_status.empty()
            st.rerun()
    elif index_exists:
        st.markdown("🗂️ Embedding file found on disk, but not loaded. Please load the embedding to use the agents!")
        with st.spinner("Loading embeddings..."):
            from langchain_huggingface import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"}
            )
            st.session_state.vector_store = FAISS.load_local(
                folder_path=index_path,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            st.rerun()
        if st.button("🔄 Regenerate embedding"):
            embedding_status.info("🔄 Processing and embedding your RAG data... This might take a moment! ⏳")
            generate_and_save_embedding(index_path)
            embedding_status.empty()
            st.rerun()
    else:
        st.markdown("⚠️ No embedding found. Please create the embedding to use the agents!")
    
        if st.button("🚀 Generate embedding"):
            embedding_status.info("🔄 Processing and embedding your RAG data... This might take a moment! ⏳")
            generate_and_save_embedding(index_path)
            embedding_status.empty()
            st.rerun()

    st.markdown("---")  # Add a separator for better visual organization
    
    # Check if CLASS is already installed
    st.markdown("### 🔧 CLASS Setup")
    if st.checkbox("Check CLASS installation status"):
        try:
            # Use sys.executable to run a simple test to see if classy can be imported
            result = subprocess.run(
                [sys.executable, "-c", "from classy import Class; print('CLASS successfully imported!')"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                st.success("✅ CLASS is already installed and ready to use!")
            else:
                st.error("❌ The 'classy' module is not installed. Please install CLASS using the button below.")
                if result.stderr:
                    st.code(result.stderr, language="bash")
        except Exception as e:
            st.error(f"❌ Error checking CLASS installation: {str(e)}")
    
    # Add CLASS installation and testing buttons
    st.text("If not installed, install CLASS to enable code execution and plotting")
    if st.button("🔄 Install CLASS"):
        # Show simple initial message
        status_placeholder = st.empty()
        status_placeholder.info("Installing CLASS... This could take a few minutes.")
        
        try:
            # Get the path to install_classy.sh
            install_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'install_classy.sh')
            
            # Make the script executable
            os.chmod(install_script_path, 0o755)
            
            # Run the installation script with shell=True to ensure proper execution
            process = subprocess.Popen(
                [install_script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                shell=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            # Create a placeholder for the current line
            current_line_placeholder = st.empty()
            
            # Collect output in the background while showing just the last line
            output_text = ""
            for line in iter(process.stdout.readline, ''):
                output_text += line
                # Update the placeholder with just the current line (real-time feedback)
                if line.strip():  # Only update for non-empty lines
                    current_line_placeholder.info(f"Current: {line.strip()}")
            
            # Get the final return code
            return_code = process.wait()
            
            # Clear the current line placeholder when done
            current_line_placeholder.empty()
            
            # Update status based on result
            if return_code == 0:
                status_placeholder.success("✅ CLASS installed successfully!")
            else:
                status_placeholder.error(f"❌ CLASS installation failed with return code: {return_code}")
                
            # Display the full output in an expander (not expanded by default)
            with st.expander("View Full Installation Log", expanded=False):
                st.code(output_text)
                
        except Exception as e:
            status_placeholder.error(f"Installation failed with exception: {str(e)}")
            st.exception(e)  # Show the full exception for debugging

    # Add test environment button
    st.text("If CLASS is installed, test the environment")
    if st.button("🧪 Test CLASS"):
        # Show simple initial message
        status_placeholder = st.empty()
        status_placeholder.info("Testing CLASS environment... This could take a moment.")
        
        try:
            # Get the path to test_classy.py
            test_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_classy.py')
            
            # Create a temporary directory for the test
            with tempfile.TemporaryDirectory() as temp_dir:
                # Run the test script with streaming output
                process = subprocess.Popen(
                    [sys.executable, test_script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=temp_dir
                )
                
                # Create a placeholder for the current line
                current_line_placeholder = st.empty()
                
                # Collect output in the background while showing just the last line
                output_text = ""
                for line in iter(process.stdout.readline, ''):
                    output_text += line
                    # Update the placeholder with just the current line (real-time feedback)
                    if line.strip():  # Only update for non-empty lines
                        current_line_placeholder.info(f"Current: {line.strip()}")
                
                # Get the final return code
                return_code = process.wait()
                
                # Clear the current line placeholder when done
                current_line_placeholder.empty()
                
                # Update status based on result
                if return_code == 0:
                    status_placeholder.success("✅ CLASS test completed successfully!")
                else:
                    status_placeholder.error(f"❌ CLASS test failed with return code: {return_code}")
                
                
                # Check for common errors
                if "ModuleNotFoundError" in output_text or "ImportError" in output_text:
                    st.error("❌ Python module import error detected. Make sure CLASS is properly installed.")
                
                if "CosmoSevereError" in output_text or "CosmoComputationError" in output_text:
                    st.error("❌ CLASS computation error detected.")
                
                # Display the full output in an expander (not expanded by default)
                with st.expander("View Full Test Log", expanded=False):
                    st.code(output_text)
                    # Check if the plot was generated
                    plot_path = os.path.join(temp_dir, 'cmb_temperature_spectrum.png')
                    if os.path.exists(plot_path):
                        # Show the plot if it was generated
                        st.subheader("Generated CMB Power Spectrum")
                        st.image(plot_path, use_container_width=True)
                    else:
                        st.warning("⚠️ No plot was generated")
                    
        except Exception as e:
            status_placeholder.error(f"Test failed with exception: {str(e)}")
            st.exception(e)  # Show the full exception for debugging
    

    st.markdown("---")  # Add a separator for better visual organization
    st.session_state.debug = st.checkbox("🔍 Show Debug Info")
    if st.button("🗑️ Reset Chat"):
        st.session_state.clear()
        st.rerun()

    if st.session_state.last_token_count > 0:
        st.markdown(f"🧮 **Last response token usage:** `{st.session_state.last_token_count}` tokens")

    # --- Display all saved plots in sidebar ---
    if "generated_plots" in st.session_state and st.session_state.generated_plots:
        with st.expander("📊 Plot Gallery", expanded=False):
            st.write("All plots generated during this session:")
            # Use a single column layout for the sidebar
            for i, plot_path in enumerate(st.session_state.generated_plots):
                if os.path.exists(plot_path):
                    st.image(plot_path, width=250, caption=os.path.basename(plot_path))
                    st.markdown("---")  # Add separator between plots

# --- Retrieval + Prompt Construction ---
def build_messages(context, question, system):
    system_msg = SystemMessage(content=system)
    human_msg = HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{question}")
    return [system_msg] + st.session_state.memory.messages + [human_msg]

#def build_messages_rating(context, question, answer, system):
#    system_msg = SystemMessage(content=system)
#    human_msg = HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{question}\n\nAI Answer:\n{answer}")
#    return [system_msg] + st.session_state.memory.messages + [human_msg]

#def build_messages_refinement(context, question, answer, feedback, system):
#    system_msg = SystemMessage(content=system)
#    human_msg = HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{question}\n\nAI Answer:\n{answer}\n\nReviewer Feedback:\n{feedback}")
#    return [system_msg] + st.session_state.memory.messages + [human_msg]

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


# Set up code execution environment
#temp_dir = tempfile.TemporaryDirectory()

class PlotAwareExecutor(LocalCommandLineCodeExecutor):
    def __init__(self, **kwargs):
        import tempfile
        # Create a persistent plots directory if it doesn't exist
        plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Still use a temp dir for code execution
        temp_dir = tempfile.TemporaryDirectory()
        kwargs['work_dir'] = temp_dir.name
        super().__init__(**kwargs)
        self._temp_dir = temp_dir
        self._plots_dir = plots_dir

    @contextlib.contextmanager
    def _capture_output(self):
        old_out, old_err = sys.stdout, sys.stderr
        buf_out, buf_err = io.StringIO(), io.StringIO()
        sys.stdout, sys.stderr = buf_out, buf_err
        try:
            yield buf_out, buf_err
        finally:
            sys.stdout, sys.stderr = old_out, old_err

    def execute_code(self, code: str):
        # 1) Extract code from markdown
        match = re.search(r"```(?:python)?\n(.*?)```", code, re.DOTALL)
        cleaned = match.group(1) if match else code
        cleaned = cleaned.replace("plt.show()", "")
        
        # Add timestamp for saving figures only if there's plt usage in the code
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        plot_filename = f'plot_{timestamp}.png'
        plot_path = os.path.join(self._plots_dir, plot_filename)
        temp_plot_path = None
        
        for line in cleaned.split("\n"):
            if "plt.savefig" in line:
                leading_spaces = line[:len(line) - len(line.lstrip())]  # get leading whitespace
                temp_plot_path = os.path.join(self._temp_dir.name, f'temporary_{timestamp}.png')
                new_line = f"{leading_spaces}plt.savefig('{temp_plot_path}', dpi=300)"
                cleaned = cleaned.replace(line, new_line)
                break
                    #else:
            # If there's a plot but no save, auto-insert save
            #    if "plt." in cleaned:
            #        temp_plot_path = os.path.join(self._temp_dir.name, f'temporary_{timestamp}.png')
            #        cleaned += f"\nplt.savefig('{temp_plot_path}')"

        # Create a temporary Python file to execute
        temp_script_path = os.path.join(self._temp_dir.name, f'temp_script_{timestamp}.py')
        with open(temp_script_path, 'w') as f:
            f.write(cleaned)
        
        full_output = ""
        try:
            # 2) Capture stdout using subprocess
            process = subprocess.Popen(
                [sys.executable, temp_script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1, 
                cwd=self._temp_dir.name
            )
            stdout, _ = process.communicate()

            # 3) Format the output
            with self._capture_output() as (out_buf, err_buf):
                if stdout:
                    out_buf.write(stdout)
                stdout_text = out_buf.getvalue()
                stderr_text = err_buf.getvalue()

            if stdout_text:
                full_output += f"STDOUT:\n{stdout_text}\n"
            if stderr_text:
                full_output += f"STDERR:\n{stderr_text}\n"
                
            # Copy plot from temp to persistent location if it exists
            if temp_plot_path and os.path.exists(temp_plot_path):
                import shutil
                shutil.copy2(temp_plot_path, plot_path)
                # Initialize the plots list if it doesn't exist
                if "generated_plots" not in st.session_state:
                    st.session_state.generated_plots = []
                # Add the persistent plot path to session state
                st.session_state.generated_plots.append(plot_path)

        except Exception:
            with self._capture_output() as (out_buf, err_buf):
                import traceback
                traceback.print_exc(file=sys.stderr)
                full_output += f"STDERR:\n{err_buf.getvalue()}\n"

        return full_output, plot_path

# Example instantiation:
executor = PlotAwareExecutor(timeout=10)

# Global agent configurations



def review_reply(feedback: Annotated[str,"Feedback on improving this reply to be accurate and relavant for the user prompt"]
                  , rating: Annotated[int,"The rating of the reply on a scale of 1 to 10"], context_variables: ContextVariables) -> ReplyResult:
    """Review the reply of the Ai Agent to the user prompt with respect to correctness, clarity and relevance for the user prompt"""
    

    
    context_variables["feedback"] = feedback
    context_variables["rating"] = rating
    context_variables["revisions"] += 1

    
    messages = list(class_agent.chat_messages.values())[0]


    #st.markdown(messages[-2])
    reply = None
    for item in messages:
        if item['name'] == 'class_agent' or item['name'] == 'improve_reply_agent':
            reply = item["content"]
       
            #  st.markdown("Last message from agent:")
            #  st.markdown(reply)

    if reply:
        context_variables["last_answer"] = reply
   
    
    if rating < 8 and context_variables["revisions"] < 3:

        return ReplyResult(
            context_variables=context_variables,
            target=AgentNameTarget("improve_reply_agent"),
            message=f'Please revise the answer considering this feedback {feedback}',
        )
    elif rating >= 8:
        #st.markdown("Formatting final answer...")

        return ReplyResult(
            context_variables=context_variables,
            target=AgentNameTarget("improve_reply_agent_final"),
            message=f'The answer is already of sufficient quality. Focus on formatting the reply',
        )
    else:
        return ReplyResult(
            context_variables=context_variables,
            target=AgentNameTarget("improve_reply_agent_final"),
            message=f'Please revise the answer considering this feedback {feedback}',
        )






if st.session_state.selected_model in GPT_MODELS:

    initial_config = LLMConfig(
        api_type="openai", 
        model=st.session_state.selected_model,
        temperature=0.2,  # Low temperature for consistent initial responses
        api_key=api_key,
    )

    review_config = LLMConfig(
        api_type="openai", 
        model=st.session_state.selected_model, 
        temperature=0.5,  # Higher temperature for creative reviews
        api_key=api_key,
        )

    #formatting_config = LLMConfig(
    #    api_type="openai", 
    #    model=st.session_state.selected_model, 
    #    temperature=0.1,  # Moderate temperature for formatting
    #    api_key=api_key,
    #)

    #code_execution_config = LLMConfig(
    #    api_type="openai", 
    #    model=st.session_state.selected_model, 
    #    temperature=0.1,  # Very low temperature for code execution
    #    api_key=api_key,
    #)

    # Global agent instances with updated system messages
    class_agent = ConversableAgent(
        name="class_agent",
        system_message=Initial_Agent_Instructions,
        description="Initial agent that answers user prompt. Expert in the CLASS code",
        human_input_mode="NEVER",
        llm_config=initial_config
    )

    review_agent = ConversableAgent(
        name="review_agent",    
        update_agent_state_before_reply=[
            UpdateSystemMessage(Review_Agent_Instructions),  # Inject the context variables into the system message, here we inject the user query to keep the review focuse
        ],
        human_input_mode="NEVER",

        description="Reviews the AI answer to user prompt",
        llm_config=review_config,
        functions=review_reply,
    )

    refine_agent = ConversableAgent(
        name="improve_reply_agent",    
        update_agent_state_before_reply=[
            UpdateSystemMessage(Refine_Agent_Instructions),  # Inject the context variables into the system message, here we inject the user query to keep the review focuse
        ],
        human_input_mode="NEVER",

        description="Improves the AI reply by taking into account the feedback",
        llm_config=initial_config,   
    )

    refine_agent_final = ConversableAgent(
        name="improve_reply_agent_final",    
        update_agent_state_before_reply=[
            UpdateSystemMessage(Refine_Agent_Instructions),  # Inject the context variables into the system message, here we inject the user query to keep the review focuse
        ],
        human_input_mode="NEVER",

        description="Improves the AI reply by taking into account the feedback",
        llm_config=initial_config,   
    )

    #formatting_agent = ConversableAgent(
    #    name="formatting_agent",    
    #    update_agent_state_before_reply=[
    #        UpdateSystemMessage(Formatting_Agent_Instructions),  # We inject the text to format directly into system message
    #    ],
    #    human_input_mode="NEVER",
    #
    #    description="Formats the final reply for the user",
    #    llm_config=formatting_config
    #)

    # no other handoffs needed as rest will be determined by function call
    class_agent.handoffs.set_after_work(AgentTarget(review_agent))
    review_agent.handoffs.set_after_work(AgentTarget(refine_agent))
    refine_agent.handoffs.set_after_work(AgentTarget(review_agent))
    refine_agent_final.handoffs.set_after_work(TerminateTarget())

    refine_agent.handoffs.add_llm_conditions([OnCondition(target=AgentTarget(refine_agent_final),condition=StringLLMCondition(prompt="The reply to the latest user question has been reviewd and received a favarable rating (equivalent to 7 or higher)"))])
    #review_agent.handoffs.add_context_condition(
    #    OnContextCondition(
    #        target=TerminateTarget(),
    #        condition=ExpressionContextCondition(ContextExpression("${rating} > 7 or ${revisions} > 2"))
    #    )
    #)

    #formatting_agent.handoffs.set_after_work(TerminateTarget())



    #code_executor = ConversableAgent(
    #    name="code_executor",
    #    system_message="""{Code_Execution_Agent_Instructions}""",
    #    human_input_mode="NEVER",
    #    llm_config=code_execution_config,
    #    code_execution_config={"executor": executor},
    #    max_consecutive_auto_reply=50

    #)
else:
    refine_agent_final = None

if st.session_state.selected_model in GEMINI_MODELS:
    initial_config_gai = LLMConfig(
        api_type="google", 
        model=st.session_state.selected_model,
        temperature=0.2,  # Low temperature for consistent initial responses
        api_key=api_key_gai,
    )

    review_config_gai = LLMConfig(
        api_type="google", 
        model=st.session_state.selected_model, 
        temperature=0.5,  # Higher temperature for creative reviews
        api_key=api_key_gai,
    )

    #formatting_config_gai = LLMConfig(
    #    api_type="google", 
    #    model=st.session_state.selected_model, 
    #    temperature=0.1,  # Moderate temperature for formatting
    #    api_key=api_key_gai,
    #)

    #code_execution_config_gai = LLMConfig(
    #    api_type="google", 
    #    model=st.session_state.selected_model, 
    #    temperature=0.1,  # Very low temperature for code execution
    #    api_key=api_key_gai,
    #)

    # Global agent instances with updated system messages for gemini
    class_agent_gai = ConversableAgent(
        name="class_agent",
        system_message=Initial_Agent_Instructions,

        description="Initial agent that answers user prompt. Expert in the CLASS code",
        human_input_mode="NEVER",
        llm_config=initial_config_gai
    )

    refine_agent_gai = ConversableAgent(
        name="improve_reply_agent",    
        update_agent_state_before_reply=[
            UpdateSystemMessage(Refine_Agent_Instructions),  # Inject the context variables into the system message, here we inject the user query to keep the review focuse
        ],
        human_input_mode="NEVER",

        description="Improves the AI reply by taking into account the feedback",
        llm_config=initial_config_gai,   
    )

    review_agent_gai = ConversableAgent(
        name="review_agent",    
        update_agent_state_before_reply=[
            UpdateSystemMessage(Review_Agent_Instructions),  # Inject the context variables into the system message, here we inject the user query to keep the review focuse
        ],
        human_input_mode="NEVER",

        description="Reviews the AI answer to user prompt",
        llm_config=review_config_gai,
        #functions=review_reply,   # funcktions are often not working as planned with gemini
    )


    #formatting_agent_gai = ConversableAgent(
    #    name="formatting_agent",    
    #    update_agent_state_before_reply=[
    #        UpdateSystemMessage(Formatting_Agent_Instructions),  # We inject the text to format directly into system message
    #    ],

    #    description="Formats the final reply for the user",
    #    human_input_mode="NEVER",
    #    llm_config=formatting_config_gai
    #)

    # no other handoffs needed as rest will be determined by function call
    class_agent_gai.handoffs.set_after_work(AgentTarget(review_agent_gai))
    review_agent_gai.handoffs.set_after_work(AgentTarget(refine_agent_gai))
    refine_agent_gai.handoffs.set_after_work(TerminateTarget())
    

    #code_executor_gai = ConversableAgent(
    #   name="code_executor",
    #    system_message="""{Code_Execution_Agent_Instructions}""",
    #    human_input_mode="NEVER",
    #    llm_config=code_execution_config_gai,
    #    code_execution_config={"executor": executor},
    #    max_consecutive_auto_reply=50
    #)
else:
    refine_agent_gai = None


def call_ai(context, user_input):
    if st.session_state.mode_is_fast == "Fast Mode":
        messages = build_messages(context, user_input, Initial_Agent_Instructions)
        response = []
        for chunk in st.session_state.llm.stream(messages):
            response.append(chunk.content)  # or chunk if using token chunks

        response = "".join(response)

        # Check if the answer contains code
        #if "```python" in response:
            # Add a note about code execution
        #    response += "\n\n> 💡 **Note**: This answer contains code. If you want to execute it, type 'execute!' in the chat."
        #    return Response(content=response)
        #else:
        #    return Response(content=response)

        return Response(content=response)
    else:
        # New Groupchat Workflow for detailed mode
        st.markdown("Thinking (Deep Thought Mode)... ")
        conversation_history = format_memory_messages(st.session_state.memory.messages)
        shared_context = ContextVariables(data =  {
            "user_prompt": user_input,
            "last_answer": "see chat history",
            "feedback": "see chat history",
            "rating": 0,
            "revisions": 0,
        })
        if st.session_state.selected_model in GEMINI_MODELS:
            pattern = AutoPattern(
                initial_agent=class_agent_gai,  # Agent that starts the conversation
                agents=[class_agent_gai,review_agent_gai,refine_agent_gai],
                group_manager_args={"llm_config": initial_config_gai},
                context_variables=shared_context,
            )
        else:
            pattern = AutoPattern(
                initial_agent=class_agent,  # Agent that starts the conversation
                agents=[class_agent,review_agent,refine_agent,refine_agent_final],
                group_manager_args={"llm_config": initial_config},
                context_variables=shared_context,
            )
        st.markdown("Generating answer...")
        result, context_variables, last_agent = initiate_group_chat(
            pattern=pattern,
            messages=f"Context from documents: {context}\n\nConversation history:\n{conversation_history}\n\nUser question: {user_input}",
            max_rounds=10,
        )
        formatted_answer = None  # default to nothing

        # 1. If the formatting agent gave the last reply, use that
        if last_agent == refine_agent_final or last_agent == refine_agent_gai:
            formatted_answer = result.chat_history[-1]["content"]

    
        # 2. Otherwise, use shared_context["last_answer"] if it's non-empty
        if not formatted_answer and shared_context.get("last_answer"):
            formatted_answer = shared_context["last_answer"]
                   
        # 3. Otherwise, fall back to the initial agent's last message
    
        if not formatted_answer:
            try:
                for item in result.chat_history:
                    st.markdown(item)
                    if item['name'] == 'class_agent' or item['name'] == 'imporve_reply_agent':
                        formatted_answer = item["content"]
            except:
                formatted_answer = 'failed to load chat history'
        

        if st.session_state.debug:
            st.session_state.debug_messages.append(("Formatted Answer", formatted_answer))
            st.session_state.debug_messages.append(("Feedback", shared_context["feedback"]))


        # Check if the answer contains code
        #if "```python" in formatted_answer:
            # Add a note about code execution
        #    formatted_answer += "\n\n> 💡 **Note**: This answer contains code. If you want to execute it, type 'execute!' in the chat."
        #    return Response(content=formatted_answer)
        #else:
        #    return Response(content=formatted_answer)
        return Response(content=formatted_answer)
def call_code():
    #if st.session_state.selected_model in GEMINI_MODELS:
    #    st.markdown("Code execution only supprted in openai at the moment")
    #    response = Response(content="Cannot excecute code with gemini api")
    #else:
        # Find the last assistant message containing code
    last_assistant_message = None
    for message in reversed(st.session_state.messages):
        if message["role"] == "assistant" and "```" in message["content"]:
            last_assistant_message = message["content"]
            break
    
    if last_assistant_message:
        st.markdown("Executing code...")
        st.info("🚀 Executing cleaned code...")
        #chat_result = code_executor.initiate_chat(
        #    recipient=code_executor,
        #    message=f"Please execute this code:\n{last_assistant_message}",
        #    max_turns=1,
        #    summary_method="last_msg"
        #)
        #execution_output = chat_result.summary
        execution_output, plot_path = executor.execute_code(last_assistant_message)
        st.subheader("Execution Output")
        st.text(execution_output)  # now contains both STDOUT and STDERR
        
        if os.path.exists(plot_path):
            st.success("✅ Plot generated successfully!")
            # Display the plot
            #st.image(plot_path, use_container_width=True)
            st.image(plot_path, width=700)
        else:
            st.warning("⚠️ No plot was generated")
        
        has_errors = any(error_indicator in execution_output for error_indicator in ["Traceback", "Error:", "Exception:", "TypeError:", "ValueError:", "NameError:", "SyntaxError:", "Error in Class"])

        
        # Check for errors and iterate if needed
        max_iterations = 3  # Maximum number of iterations to prevent infinite loops
        current_iteration = 0
        
        while has_errors and current_iteration < max_iterations:
            current_iteration += 1
            st.error(f"Previous error: {execution_output}")  # Show the actual error message
            st.info(f"🔧 Fixing errors (attempt {current_iteration}/{max_iterations})...")

            # Get new review with error information
            #review_message = f"""
            #Previous answer had errors during execution:
            #{execution_output}

            #Please review and suggest fixes for this answer. IMPORTANT: Preserve all code blocks exactly as they are, only fix actual errors:
            #{last_assistant_message}
            #"""
            #chat_result_2 = review_agent.initiate_chat(
            #    recipient=review_agent,
            #    message=review_message,
            #    max_turns=1,
            #    summary_method="last_msg"
            #)
            #review_feedback = chat_result_2.summary
            #if st.session_state.debug:
            #    st.session_state.debug_messages.append(("Error Review Feedback", review_feedback))

            # Get corrected version
            #chat_result_3 = initial_agent.initiate_chat(
            #    recipient=initial_agent,
            #    message=f"""Original answer: {last_assistant_message}
            #    Review feedback with error fixes: {review_feedback}
            #    IMPORTANT: Only fix actual errors in the code blocks. Preserve all working code exactly as it is.""",
            #    max_turns=1,
            #    summary_method="last_msg"
            #)
            #corrected_answer = chat_result_3.summary
            #if st.session_state.debug:
            #    st.session_state.debug_messages.append(("Corrected Answer", corrected_answer))

            # Format the corrected answer
            #chat_result_4 = formatting_agent.initiate_chat(
            #    recipient=formatting_agent,
            #    message=f"""Please format this corrected answer while preserving all code blocks:
            #    {corrected_answer}
            #    """,
            #    max_turns=1,
            #    summary_method="last_msg"
            #)
            #formatted_answer = chat_result_4.summary
            #if st.session_state.debug:
            #    st.session_state.debug_messages.append(("Formatted Corrected Answer", formatted_answer))

            # get context on error message
            context = retrieve_context(execution_output)

            review_message = f"""
            Context:\n{context}\n\nQuestion:

            Previous answer had errors during execution:
            {execution_output}

            Please modify the code to fix those errors. IMPORTANT: Preserve all code blocks exactly as they are, only fix actual errors:
            {last_assistant_message}
            """


            # initialise context to update agent messages
            shared_context = ContextVariables(data =  {
                "user_prompt": "Correct the errors in the code",
                "last_answer": last_assistant_message,
                "feedback": f" Previous answer had errors during execution: {execution_output}",
                "rating": 0,
                "revisions": 0,
            })

            if st.session_state.selected_model in GEMINI_MODELS:
                pattern = AutoPattern(
                    initial_agent=refine_agent_gai,  # Agent that starts the conversation
                    agents=[refine_agent_gai],
                    group_manager_args={"llm_config": initial_config_gai},
                    context_variables=shared_context,
                )
            else:
                pattern = AutoPattern(
                    initial_agent=refine_agent_final,  # Agent that starts the conversation
                    agents=[refine_agent_final],
                    group_manager_args={"llm_config": initial_config},
                    context_variables=shared_context,
                )
            
            result, context_variables, last_agent = initiate_group_chat(
                pattern=pattern,
                messages=review_message,
                max_rounds=2,
            )

            #if st.session_state.selected_model in GEMINI_MODELS:
            #    chat_result = review_agent_gai.initiate_chat(
            #        recipient=refine_agent_gai,
            #        message=review_message,
            #        max_turns=1,
            #        summary_method="last_msg"
            #    )
            #else:
            #    chat_result = review_agent.initiate_chat(
            #        recipient=refine_agent,
            #        message=review_message,
            #        max_turns=1,
            #        summary_method="last_msg"
            #    )

            formatted_answer = result.chat_history[-1]["content"]
            if st.session_state.debug:
                st.session_state.debug_messages.append(("Error Review Feedback", formatted_answer))


            # Execute the corrected code
            st.info("🚀 Executing corrected code...")
            #chat_result = code_executor.initiate_chat(
            #    recipient=code_executor,
            #    message=f"Please execute this corrected code:\n{formatted_answer}",
            #    max_turns=1,
            #    summary_method="last_msg"
            #)
            #execution_output = chat_result.summary
            execution_output, plot_path = executor.execute_code(formatted_answer)
            st.subheader("Execution Output")
            st.text(execution_output)  # now contains both STDOUT and STDERR
            
            if os.path.exists(plot_path):
                st.success("✅ Plot generated successfully!")
                # Display the plot
                st.image(plot_path, width=700)
            else:
                st.warning("⚠️ No plot was generated")
            
            if st.session_state.debug:
                st.session_state.debug_messages.append(("Execution Output", execution_output))
            
            # If we've reached the end of iterations and we're successful
            #if not has_errors or current_iteration == max_iterations:
                # Add successful execution to the conversation with plot
            #    final_answer = formatted_answer if formatted_answer else last_assistant_message
            #    response_text = f"Execution completed successfully:\n{execution_output}\n\nThe following code was executed:\n```python\n{final_answer}\n```"
                
            #    # Add plot path marker for rendering in the conversation
            #    if os.path.exists(plot_path):
            #        response_text += f"\n\nPLOT_PATH:{plot_path}\n"
                    
            #    if current_iteration > 0:
            #        response_text = f"After {current_iteration} correction attempts: " + response_text
                
                # Set the response variable with our constructed text that includes plot
            #   response = Response(content=response_text)
            
            # Update last_assistant_message with the formatted answer for next iteration
            last_assistant_message = formatted_answer
            has_errors = any(error_indicator in execution_output for error_indicator in ["Traceback", "Error:", "Exception:", "TypeError:", "ValueError:", "NameError:", "SyntaxError:", "Error in Class"])

        if has_errors:
            st.markdown("> ⚠️ **Note**: Some errors could not be fixed after multiple attempts. You can request changes by describing them in the chat.")
            st.markdown(f"> ❌ Last execution message:\n{execution_output}")

            # Display the final code that was successfully executed
            with st.expander("View Failed Code", expanded=False):
                st.markdown(last_assistant_message)
            response = Response(content=f"Execution completed with errors:\n{execution_output}\n\nThe following code was executed:\n```python\n{last_assistant_message}\n")
        else:
            # Check for common error indicators in the output
            if any(error_indicator in execution_output for error_indicator in ["Traceback", "Error:", "Exception:", "TypeError:", "ValueError:", "NameError:", "SyntaxError:"]):
                st.markdown("> ⚠️ **Note**: Code execution completed but with errors. You can request changes by describing them in the chat.")
                st.markdown(f"> ❌ Execution message:\n{execution_output}")
                
                    # Display the final code that was successfully executed
                with st.expander("View Failed Code", expanded=False):
                    st.markdown(last_assistant_message)
                response = Response(content=f"Execution completed with errors:\n{execution_output}\n\nThe following code was executed:\n```python\n{last_assistant_message}\n")

            else:
                st.markdown(f"> ✅ Code executed successfully. Last execution message:\n{execution_output}")
                
                # Display the final code that was successfully executed
                with st.expander("View Successfully Executed Code", expanded=False):
                    st.markdown(last_assistant_message)
                    
                # Create a response message that includes the plot path
                response_text = f"Execution completed successfully:\n{execution_output}\n\nThe following code was executed:\n```python\n{last_assistant_message}\n```"
                
                # Add plot path marker for rendering in the conversation
                if os.path.exists(plot_path):
                    response_text += f"\n\nPLOT_PATH:{plot_path}\n"
                    
                response = Response(content=response_text)
    else:
        response = Response(content="No code found to execute in the previous messages.")

    return response
# --- Chat Input ---
if OPTIONS:
    user_input = st.chat_input("Type your prompt here...")
else:
    user_input = None

# --- Display Full Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Check if this message contains a plot path marker
        if "PLOT_PATH:" in message["content"]:
            # Split content into text and plot path
            parts = message["content"].split("PLOT_PATH:")
            # Display the text part
            st.markdown(parts[0])
            # Display each plot path
            for plot_info in parts[1:]:
                plot_path = plot_info.split('\n')[0].strip()
                if os.path.exists(plot_path):
                    st.image(plot_path, width=700)
        else:
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
        
        # Initialization with streaming
       
        if st.session_state.mode_is_fast == "Fast Mode":

            if st.session_state.selected_model in GEMINI_MODELS:


                st.session_state.llm = ChatGoogleGenerativeAI(
                        model=st.session_state.selected_model,
                        #streaming=True,
                        callbacks=[stream_handler],
                        google_api_key=api_key_gai,
                        temperature=0.2,
                        convert_system_message_to_human=True  # Important for compatibility
                )


            else:
                st.session_state.llm = ChatOpenAI(
                        model_name=st.session_state.selected_model,
                        streaming=True,
                        callbacks=[stream_handler],
                        openai_api_key=api_key,
                        temperature=0.2
                )


        # Check if this is an execution request
        if user_input.strip().lower() == "execute!" or user_input.strip().lower() == "plot!":
            response = call_code()

        else:
            response = call_ai(context, user_input)
            if st.session_state.mode_is_fast != "Fast Mode":
                st.markdown(response.content)

        st.session_state.memory.add_ai_message(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response.content})



# --- Display Welcome Message (outside of sidebar) ---
# This ensures the welcome message appears in the main content area
if "llm_initialized" in st.session_state and st.session_state.llm_initialized and st.session_state.vector_store and not st.session_state.greeted:
    # Create a chat message container for the welcome message
    with st.chat_message("assistant"):
        # Create empty container for streaming
        welcome_container = st.empty()
        
        # Set up the streaming handler
        welcome_stream_handler = StreamHandler(welcome_container)


        if st.session_state.selected_model in GEMINI_MODELS:

            #streaming_llm = genai.GenerativeModel(model_name=st.session_state.selected_model)

            streaming_llm = ChatGoogleGenerativeAI(
                model=st.session_state.selected_model,
                google_api_key=api_key_gai,
                #streaming=True,
                callbacks=[welcome_stream_handler],
                temperature=1.0,
                convert_system_message_to_human=True  # Important for compatibility
            )
        
        else:
        # Initialize streaming LLM
            streaming_llm = ChatOpenAI(
                model_name=st.session_state.selected_model,
                streaming=True,
                callbacks=[welcome_stream_handler],
                openai_api_key=api_key,
                temperature=1.0
            )
        
        # Generate the streaming welcome message
        messages = [
            SystemMessage(content=Initial_Agent_Instructions),
            HumanMessage(content="Please greet the user and briefly explain what you can do as the CLASS code assistant.")
        ]

        greeting = streaming_llm.invoke(messages)
        
        # Save the completed message to history
        st.session_state.messages.append({"role": "assistant", "content": greeting.content})
        st.session_state.memory.add_ai_message(greeting.content)
        st.session_state.greeted = True

        if st.session_state.selected_model in GEMINI_MODELS:
            # non streaming greeting
            st.markdown(greeting.content)


# --- Debug Info ---
if st.session_state.debug:
    with st.sidebar.expander("🛠️ Debug Information", expanded=True):
        # Create a container for debug messages
        debug_container = st.container()
        with debug_container:
            st.markdown("### Debug Messages")
            
            # Display all debug messages in a scrollable container
            for title, message in st.session_state.debug_messages:
                st.markdown(f"### {title}")
                st.markdown(message)
                st.markdown("---")
    
    with st.sidebar.expander("🛠️ Context Used"):
        if "context" in locals():
            st.markdown(context)
        else:
            st.markdown("No context retrieved yet.")

if st.sidebar.button("execute code"):
    response = call_code()
    st.session_state.memory.add_ai_message(response.content)
    st.session_state.messages.append({"role": "assistant", "content": response.content})


# Utility function to gather all docs from class-data
