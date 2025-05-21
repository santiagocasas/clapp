# This script requires Streamlit and LangChain
# Install it with: pip install streamlit openai langchain langchain-openai langchain-community

import streamlit as st
import time
import json
import os
import base64
import getpass
from cryptography.fernet import Fernet
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document

from langchain.callbacks.base import BaseCallbackHandler

from pydantic import BaseModel, Field
from typing import Annotated


from autogen import ConversableAgent, LLMConfig, UpdateSystemMessage

from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import AutoPattern
from autogen.agentchat.group import ReplyResult, AgentNameTarget
from autogen.agentchat.group import AgentTarget, RevertToUserTarget, TerminateTarget, NestedChatTarget
from autogen.agentchat.group import OnCondition, StringLLMCondition
from autogen.agentchat.group import OnContextCondition, ExpressionContextCondition, ContextExpression
from typing import Any, Annotated
from autogen.agentchat.group import ContextVariables


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

# --- Helper Functions ---
def save_encrypted_key(encrypted_key, username):
    """Save encrypted key to file with username prefix"""
    try:
        filename = f"{username}_encrypted_api_key" if username else ".encrypted_api_key"
        with open(filename, "w") as f:
            f.write(encrypted_key)
        return True
    except Exception as e:
        return False

def load_encrypted_key(username):
    """Load encrypted key from file with username prefix"""
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


class Feedback(BaseModel):
    grade: Annotated[int, Field(description="Score from 1 to 10")]
    improvement_instructions: Annotated[str, Field(description="Advice on how to improve the reply")]

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

def inject_global_styles_and_font(font_name: str):
    font_url_name = font_name.replace(" ", "+")
    st.markdown(
        f"""
        <link href="https://fonts.googleapis.com/css?family={font_url_name}" rel="stylesheet">
        <style>
        .main-title {{
            font-family: '{font_name}', sans-serif !important;
            font-size: 3.8rem !important;
            font-weight: 700 !important;
            margin-bottom: 0.5rem !important;
            margin-top: 0.5rem !important;
            color: #222;
            letter-spacing: 1px;
        }}
        .app-title, .section4-header {{
            font-family: '{font_name}', sans-serif !important;
            font-size: 2rem !important;
            font-weight: 600 !important;
            margin-bottom: 0.2rem !important;
            margin-top: 0.5rem !important;
            color: #222;
            letter-spacing: 0.5px;
        }}
        .section4-header {{
            margin-bottom: -0.5rem !important;
        }}
        div[data-testid="stSelectbox"] {{
            margin-top: -0.5rem !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call this ONCE at the top
inject_global_styles_and_font("Jersey 10")

st.markdown('<div class="main-title">CLAPP: CLASS LLM Agent for Pair Programming</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("images/CLAPP.png", width=400)


# New prompts for the swarm
Initial_Agent_Instructions = read_prompt_from_file("prompts/class_instructions.txt") # Reuse or adapt class_instructions
Review_Agent_Instructions = read_prompt_from_file("prompts/review_instructions.txt") # Adapt rating_instructions
#Typo_Agent_Instructions = read_prompt_from_file("prompts/typo_instructions.txt")   # New prompt file
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
    if "debug_messages" not in st.session_state:
        st.session_state.debug_messages = []


init_session()



# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown('<div class="app-title">🔐 API & Assistants</div>', unsafe_allow_html=True)
    api_key = st.text_input("1. OpenAI API Key", type="password")
    username = st.text_input("2. Username (for saving your API key)", placeholder="Enter your username")
    user_password = st.text_input("3. Password to encrypt/decrypt API key", type="password")
    
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
                if save_encrypted_key(encrypted_key.decode(), username):
                    st.success("API key encrypted and saved! ✅")
                else:
                    st.warning("API key encrypted but couldn't save to file! ⚠️")
            except Exception as e:
                st.error(f"Error saving API key: {str(e)}")
    
    # Try to load saved API key if password is provided
    elif user_password and not api_key:
        # Try to load from file first
        encrypted_key = load_encrypted_key(username)
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
    if st.button("🗑️ Clear Saved API Key"):
        deleted_files = False
        error_message = ""
        
        # Try to delete username-specific file if it exists
        if username:
            filename = f"{username}_encrypted_api_key"
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                    deleted_files = True
                    st.success(f"Deleted key file for user: {username}")
                except Exception as e:
                    error_message += f"Error clearing {filename}: {str(e)}\n"
        
        # Also try to delete the default file if it exists
        if os.path.exists(".encrypted_api_key"):
            try:
                os.remove(".encrypted_api_key")
                deleted_files = True
                st.success("Deleted default key file")
            except Exception as e:
                error_message += f"Error clearing default key file: {str(e)}\n"
        
        # Clean up session state
        if "saved_api_key" in st.session_state:
            del st.session_state.saved_api_key
        if "encrypted_key" in st.session_state:
            del st.session_state.encrypted_key
        
        # Show appropriate message
        if deleted_files:
            st.info("Session cleared. Reloading page...")
            time.sleep(1)  # Brief pause so user can see the message
            st.rerun()
        elif error_message:
            st.error(error_message)
        else:
            st.warning("No saved API keys found to delete.")

    st.markdown('<div class="app-title">4. Choose LLM model 🧠</div>', unsafe_allow_html=True)

    st.session_state.selected_model = st.selectbox(
        "",  # No label, since we use our own
        options=["gpt-4o-mini", "gpt-4o"],
        index=["gpt-4o-mini", "gpt-4o"].index(st.session_state.selected_model)
    )


    # Check if model has changed
    if "previous_model" not in st.session_state:
        st.session_state.previous_model = st.session_state.selected_model
    elif st.session_state.previous_model != st.session_state.selected_model:
        # Reset relevant state variables when model changes
        st.session_state.vector_store = None
        st.session_state.greeted = False
        st.session_state.messages = []
        st.session_state.memory = ChatMessageHistory()
        st.session_state.previous_model = st.session_state.selected_model
        st.info("Model changed! Please initialize again with the new model.")

    st.markdown('<div class="app-title">Response Mode</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        mode_is_fast = st.toggle("Fast Mode", value=True)
    with col2:
        if mode_is_fast:
            st.caption("✨ Quick responses with good quality (recommended for most uses)")
        else:
            st.caption("🎯 Swarm mode, more refined responses (may take longer)")
    

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Initialize only after model is selected
        if st.button("🚀 Initialize with Selected Model"):
            # First initialization without streaming
            st.session_state.llm = ChatOpenAI(
                    model_name=st.session_state.selected_model,
                    openai_api_key=api_key,
                    temperature=1.0
            )

            if st.session_state.vector_store is None:
                embedding_status = st.empty()
                embedding_status.info("🔄 Processing and embedding your RAG data... This might take a moment! ⏳")
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
                embedding_status.empty()  # Clear the loading message

            # Initialize but don't generate welcome message yet
            if not st.session_state.greeted:
                # Just set the initialized flag, we'll generate the welcome message later
                st.session_state.llm_initialized = True
                st.rerun()  # Refresh the page to show the initialized state

    st.markdown("---")  # Add a separator for better visual organization
    
    # Check if CLASS is already installed
    st.markdown('<div class="app-title">🔧 CLASS Setup</div>', unsafe_allow_html=True)
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
                temp_plot_path = os.path.join(self._temp_dir.name, f'temporary_{timestamp}.png')
                cleaned = cleaned.replace(line, f"plt.savefig('{temp_plot_path}', dpi=300)")
                break
        else:
            # If there's a plot but no save, auto-insert save
            if "plt." in cleaned:
                temp_plot_path = os.path.join(self._temp_dir.name, f'temporary_{timestamp}.png')
                cleaned += f"\nplt.savefig('{temp_plot_path}')"

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
initial_config = LLMConfig(
    api_type="openai", 
    model=st.session_state.selected_model,
    temperature=0.2,  # Low temperature for consistent initial responses
    api_key=api_key,
)

review_config = LLMConfig(
    api_type="openai", 
    model=st.session_state.selected_model, 
    temperature=0.7,  # Higher temperature for creative reviews
    api_key=api_key,
    response_format=Feedback
)

# typo_config = LLMConfig(
#     api_type="openai", 
#     model=st.session_state.selected_model, 
#     temperature=0.1,  # Very low temperature for precise code corrections
#     api_key=api_key,
# )

formatting_config = LLMConfig(
    api_type="openai", 
    model=st.session_state.selected_model, 
    temperature=0.3,  # Moderate temperature for formatting
    api_key=api_key,
)

code_execution_config = LLMConfig(
    api_type="openai", 
    model=st.session_state.selected_model, 
    temperature=0.1,  # Very low temperature for code execution
    api_key=api_key,
)

def review_reply(reply: Annotated[str,"The reply to user prompt by an AI agent"], 
                 feedback: Annotated[str,"Feedback on improving this reply to be accurate and relavant for the user prompt"], 
                 rating: Annotated[int,"The rating of the reply on a scale of 1 to 10"], 
                 context_variables: ContextVariables) -> ReplyResult:
    """Review the reply of the AI Agent to the user prompt with respect to correctness, clarity and relevance for the user prompt"""
    

    context_variables["best_answer"] = reply
    context_variables["feedback"] = feedback
    context_variables["rating"] = rating
    context_variables["revisions"] += 1


    #st.markdown("Reviewing draft...")

    #if st.session_state.debug:

    #    st.session_state.debug_messages.append(("Orignal Answer", reply))
    #    st.session_state.debug_messages.append(("Review Feedback", feedback))

    
    if rating <= 7 and context_variables["revisions"] < 3:

        return ReplyResult(
            context_variables=context_variables,
            target=AgentTarget(initial_agent),
            message=f'Please revise your answer considering this feedback {feedback}',
        )
    else:
        #st.markdown("Formatting final answer...")

        return ReplyResult(
            context_variables=context_variables,
            target=AgentTarget(formatting_agent),
            message=f'Please formatt the following reply: {reply}',
        )

# Global agent instances with updated system messages
initial_agent = ConversableAgent(
    name="initial_agent",
    system_message=Initial_Agent_Instructions,
    human_input_mode="NEVER",
    llm_config=initial_config
)

review_agent = ConversableAgent(
    name="review_agent",    
    update_agent_state_before_reply=[
        UpdateSystemMessage(Review_Agent_Instructions),  # Inject the context variables into the system message, here we inject the user query to keep the review focuse
    ],
    human_input_mode="NEVER",
    llm_config=review_config,
    functions=review_reply,
)

# typo_agent = ConversableAgent(
#     name="typo_agent",
#     system_message=f"""You are the typo and code correction agent. Your task is to:
# 1. Fix any typos or grammatical errors
# 2. Correct any code issues
# 3. Ensure proper formatting
# 4. Maintain the original meaning while improving clarity
# 5. Verify plots are saved to disk (not using show())
# 6. PRESERVE all code blocks exactly as they are unless there are actual errors
# 7. If no changes are needed, keep the original code blocks unchanged

# # {Typo_Agent_Instructions}""",
# #     human_input_mode="NEVER",
# #     llm_config=typo_config
# # )

formatting_agent = ConversableAgent(
    name="formatting_agent",    
    update_agent_state_before_reply=[
        UpdateSystemMessage(Formatting_Agent_Instructions),  # We inject the text to format directly into system message
    ],
    human_input_mode="NEVER",
    llm_config=formatting_config
)

# no other handoffs needed as rest will be determined by function call
initial_agent.handoffs.set_after_work(AgentTarget(review_agent))
formatting_agent.handoffs.set_after_work(TerminateTarget())



code_executor = ConversableAgent(
    name="code_executor",
    system_message="""{Code_Execution_Agent_Instructions}""",
    human_input_mode="NEVER",
    llm_config=code_execution_config,
    code_execution_config={"executor": executor},
    max_consecutive_auto_reply=50
)

def call_ai(context, user_input):
    if mode_is_fast:
        messages = build_messages(context, user_input, Initial_Agent_Instructions)
        response = st.session_state.llm.invoke(messages)
        return Response(content=response.content)
    else:
        # New Groupchat Workflow for detailed mode
        st.markdown("Thinking (Deep Thought Mode)... ")

        # Format the conversation history for context
        conversation_history = format_memory_messages(st.session_state.memory.messages)

        shared_context = ContextVariables(data =  {
            "user_prompt": user_input,
            "best_answer": "",
            "feedback": "",
            "rating": None,
            "revisions": 0,
        })

        pattern = AutoPattern(
            initial_agent=initial_agent,  # Agent that starts the conversation
            agents=[initial_agent,review_agent,formatting_agent],
            group_manager_args={"llm_config": initial_config},
            context_variables=shared_context,
        )


        st.markdown("Generating answer...")

        result, context_variables, last_agent = initiate_group_chat(
            pattern=pattern,
            messages=f"Context from documents: {context}\n\nConversation history:\n{conversation_history}\n\nUser question: {user_input}",
            max_rounds=10,
        )

        # 1. Initial Agent generates the draft
        #st.markdown("Generating initial draft...")
        #chat_result_1 = initial_agent.initiate_chat(
        #    recipient=initial_agent,
        #    message=f"Conversation history:\n{conversation_history}\n\nContext from documents: {context}\n\nUser question: {user_input}",
        #    max_turns=1,
        #    summary_method="last_msg"
        #)
        #draft_answer = chat_result_1.summary
        #if st.session_state.debug:
        #    st.session_state.debug_messages.append(("Initial Draft", draft_answer))

        # 2. Review Agent critiques the draft
        #st.markdown("Reviewing draft...")
        #chat_result_2 = review_agent.initiate_chat(
        #    recipient=review_agent,
        #    message=f"Conversation history:\n{conversation_history}\n\nPlease review this draft answer:\n{draft_answer}",
        #    max_turns=1,
        #    summary_method="last_msg"
        #)
        #review_feedback = chat_result_2.summary
        #if st.session_state.debug:
        #    st.session_state.debug_messages.append(("Review Feedback", review_feedback))

        # # 3. Typo Agent corrects the draft
        # st.markdown("Checking for typos...")
        # chat_result_3 = typo_agent.initiate_chat(
        #     recipient=typo_agent,
        #     message=f"Original draft: {draft_answer}\n\nReview feedback: {review_feedback}",
        #     max_turns=1,
        #     summary_method="last_msg"
        # )
        # typo_corrected_answer = chat_result_3.summary
        # if st.session_state.debug: st.text(f"Typo-Corrected Answer:\n{typo_corrected_answer}")

        # 4. Formatting Agent formats the final answer
        #st.markdown("Formatting final answer...")
        #chat_result_4 = formatting_agent.initiate_chat(
        #    recipient=formatting_agent,
        #    message=f"""Please format this answer while preserving any code blocks:
        #        {draft_answer}""",
        #    max_turns=1,
        #    summary_method="last_msg"
        #)
        #formatted_answer = chat_result_4.summary
        
        formatted_answer = shared_context["best_answer"]

        if st.session_state.debug:
            st.session_state.debug_messages.append(("Formatted Answer", formatted_answer))
            st.session_state.debug_messages.append(("Feedback", shared_context["feedback"]))


        # Check if the answer contains code
        if "```python" in formatted_answer:
            # Add a note about code execution
            formatted_answer += "\n\n> 💡 **Note**: This answer contains code. If you want to execute it, type 'execute!' in the chat."
            return Response(content=formatted_answer)
        else:
            return Response(content=formatted_answer)


# --- Chat Input ---
user_input = st.chat_input("Type your prompt here...")

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

        # Second initialization with streaming
        st.session_state.llm = ChatOpenAI(
                model_name=st.session_state.selected_model,
                streaming=True,
                callbacks=[stream_handler],
                openai_api_key=api_key,
                temperature=0.2
        )

        # Check if this is an execution request
        if user_input.strip().lower() == "execute!":
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
                
                # Check for errors and iterate if needed
                max_iterations = 3  # Maximum number of iterations to prevent infinite loops
                current_iteration = 0
                has_errors = any(error_indicator in execution_output for error_indicator in ["Traceback", "Error:", "Exception:", "TypeError:", "ValueError:", "NameError:", "SyntaxError:", "Error in Class"])

                while has_errors and current_iteration < max_iterations:
                    current_iteration += 1
                    st.error(f"Previous error: {execution_output}")  # Show the actual error message
                    st.info(f"🔧 Fixing errors (attempt {current_iteration}/{max_iterations})...")

                    # Get new review with error information
                    review_message = f"""
                    Previous answer had errors during execution:
                    {execution_output}

                    Please review and suggest fixes for this answer. IMPORTANT: Preserve all code blocks exactly as they are, only fix actual errors:
                    {last_assistant_message}
                    """
                    chat_result_2 = review_agent.initiate_chat(
                        recipient=review_agent,
                        message=review_message,
                        max_turns=1,
                        summary_method="last_msg"
                    )
                    review_feedback = chat_result_2.summary
                    if st.session_state.debug:
                        st.session_state.debug_messages.append(("Error Review Feedback", review_feedback))

                    # Get corrected version
                    chat_result_3 = initial_agent.initiate_chat(
                        recipient=initial_agent,
                        message=f"""Original answer: {last_assistant_message}
                        Review feedback with error fixes: {review_feedback}
                        IMPORTANT: Only fix actual errors in the code blocks. Preserve all working code exactly as it is.""",
                        max_turns=1,
                        summary_method="last_msg"
                    )
                    corrected_answer = chat_result_3.summary
                    if st.session_state.debug:
                        st.session_state.debug_messages.append(("Corrected Answer", corrected_answer))

                    # Format the corrected answer
                    chat_result_4 = formatting_agent.initiate_chat(
                        recipient=formatting_agent,
                        message=f"""Please format this corrected answer while preserving all code blocks:
                        {corrected_answer}
                        """,
                        max_turns=1,
                        summary_method="last_msg"
                    )
                    formatted_answer = chat_result_4.summary
                    if st.session_state.debug:
                        st.session_state.debug_messages.append(("Formatted Corrected Answer", formatted_answer))

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
                    if not has_errors or current_iteration == max_iterations:
                        # Add successful execution to the conversation with plot
                        final_answer = formatted_answer if formatted_answer else last_assistant_message
                        response_text = f"Execution completed successfully:\n{execution_output}\n\nThe following code was executed:\n```python\n{final_answer}\n```"
                        
                        # Add plot path marker for rendering in the conversation
                        if os.path.exists(plot_path):
                            response_text += f"\n\nPLOT_PATH:{plot_path}\n"
                            
                        if current_iteration > 0:
                            response_text = f"After {current_iteration} correction attempts: " + response_text
                        
                        # Set the response variable with our constructed text that includes plot
                        response = Response(content=response_text)
                    
                    # Update last_assistant_message with the formatted answer for next iteration
                    last_assistant_message = formatted_answer
                    has_errors = any(error_indicator in execution_output for error_indicator in ["Traceback", "Error:", "Exception:", "TypeError:", "ValueError:", "NameError:", "SyntaxError:", "Error in Class"])

                if has_errors:
                    st.markdown("> ⚠️ **Note**: Some errors could not be fixed after multiple attempts. You can request changes by describing them in the chat.")
                    st.markdown(f"> ❌ Last execution message:\n{execution_output}")
                    response = Response(content=f"Execution completed with errors:\n{execution_output}")
                else:
                    # Check for common error indicators in the output
                    if any(error_indicator in execution_output for error_indicator in ["Traceback", "Error:", "Exception:", "TypeError:", "ValueError:", "NameError:", "SyntaxError:"]):
                        st.markdown("> ⚠️ **Note**: Code execution completed but with errors. You can request changes by describing them in the chat.")
                        st.markdown(f"> ❌ Execution message:\n{execution_output}")
                        response = Response(content=f"Execution completed with errors:\n{execution_output}")
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
        else:
            response = call_ai(context, user_input)
            if not mode_is_fast:
                st.markdown(response.content)

        st.session_state.memory.add_ai_message(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response.content})

# --- Display Welcome Message (outside of sidebar) ---
# This ensures the welcome message appears in the main content area
if "llm_initialized" in st.session_state and st.session_state.llm_initialized and not st.session_state.greeted:
    # Create a chat message container for the welcome message
    with st.chat_message("assistant"):
        # Create empty container for streaming
        welcome_container = st.empty()
        
        # Set up the streaming handler
        welcome_stream_handler = StreamHandler(welcome_container)
        
        # Initialize streaming LLM
        streaming_llm = ChatOpenAI(
            model_name=st.session_state.selected_model,
            streaming=True,
            callbacks=[welcome_stream_handler],
            openai_api_key=api_key,
            temperature=0.2
        )
        
        # Generate the streaming welcome message
        greeting = streaming_llm.invoke([
            SystemMessage(content=Initial_Agent_Instructions),
            HumanMessage(content="Please greet the user and briefly explain what you can do as the CLASS code assistant.")
        ])
        
        # Save the completed message to history
        st.session_state.messages.append({"role": "assistant", "content": greeting.content})
        st.session_state.memory.add_ai_message(greeting.content)
        st.session_state.greeted = True

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