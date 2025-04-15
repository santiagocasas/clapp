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


from autogen import ConversableAgent, LLMConfig, UpdateSystemMessage
import tempfile
from autogen.coding import LocalCommandLineCodeExecutor, CodeBlock
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
import io
from PIL import Image
import re

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

# New prompts for the swarm
Initial_Agent_Instructions = read_prompt_from_file("prompts/class_instructions.txt") # Reuse or adapt class_instructions
Review_Agent_Instructions = read_prompt_from_file("prompts/review_instructions.txt") # Adapt rating_instructions
Typo_Agent_Instructions = read_prompt_from_file("prompts/typo_instructions.txt")   # New prompt file
Formatting_Agent_Instructions = read_prompt_from_file("prompts/formatting_instructions.txt") # New prompt file

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
            st.caption("🎯 Swarm mode, more refined responses (may take longer)")
    
    st.markdown("---")  # Add a separator for better visual organization

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

def format_execution_results(execution_result, plot_buffer=None):
    """Format the execution results in a more readable way using markdown."""
    if not execution_result:
        return "No code execution results to display."
    
    # Split the result into lines for better processing
    lines = execution_result.split('\n')
    formatted_result = "### Code Execution Results\n\n"
    
    # Track if we're in a code block
    in_code_block = False
    current_code_block = []
    
    for line in lines:
        # Check for code block markers
        if line.startswith('```'):
            if in_code_block:
                # End of code block
                formatted_result += "```python\n" + '\n'.join(current_code_block) + "\n```\n\n"
                current_code_block = []
                in_code_block = False
            else:
                # Start of code block
                in_code_block = True
        elif in_code_block:
            # Inside code block
            current_code_block.append(line)
        else:
            # Regular text
            if line.strip():
                if line.startswith('Error'):
                    formatted_result += f"❌ **{line}**\n\n"
                elif line.startswith('Warning'):
                    formatted_result += f"⚠️ **{line}**\n\n"
                elif line.startswith('Success'):
                    formatted_result += f"✅ **{line}**\n\n"
                else:
                    formatted_result += f"{line}\n\n"
    
    # Handle any remaining code block
    if current_code_block:
        formatted_result += "```python\n" + '\n'.join(current_code_block) + "\n```\n\n"
    
    # Add plot if available
    if plot_buffer:
        formatted_result += "### Plot Output\n\n"
        st.image(plot_buffer, use_column_width=True)
    
    return formatted_result

def clean_code_for_execution(code):
    """Remove plt.show() calls and add plt.savefig() for proper plot capture."""
    # Split code into lines
    lines = code.split('\n')
    cleaned_lines = []
    
    # Track if we're in a code block
    in_code_block = False
    has_plotting = False
    
    for line in lines:
        # Check for code block markers
        if line.startswith('```'):
            if in_code_block:
                # End of code block, add savefig if needed
                if has_plotting:
                    cleaned_lines.append("fig = plt.gcf()")
                    cleaned_lines.append("fig.savefig('plot.png', dpi=300, bbox_inches='tight')")
                    cleaned_lines.append("plt.close('all')")
                in_code_block = False
                has_plotting = False
            else:
                in_code_block = True
            cleaned_lines.append(line)
            continue
        
        if in_code_block:
            # Skip plt.show() calls
            if 'plt.show()' in line:
                continue
            # Skip other problematic statements
            if line.strip().startswith('show('):
                continue
            # Check for plotting commands
            if any(plot_cmd in line for plot_cmd in ['plt.plot', 'plt.scatter', 'plt.bar', 'plt.hist', 'plt.imshow', 'plt.figure', 'plt.subplot']):
                has_plotting = True
            cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

# Set up code execution environment
temp_dir = tempfile.TemporaryDirectory()

class PlotAwareExecutor(LocalCommandLineCodeExecutor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.plot_buffer = None

    def execute_code(self, code: str):
        plt.close("all")

        # Extract code from triple backticks if needed
        match = re.search(r"```(?:python)?\n(.*?)```", code, re.DOTALL)
        if match:
            code = match.group(1)

        # Clean the code for execution
        cleaned_code = clean_code_for_execution(code)

        if st.session_state.debug:
            st.text(">>> FINAL EXECUTED CODE:\n" + cleaned_code)

        # Wrap as CodeBlock
        code_block = CodeBlock(language="python", code=cleaned_code)
        result = super().execute_code_blocks([code_block])

        # Capture plot
        if os.path.exists("plot.png"):
            with open("plot.png", "rb") as f:
                buf = io.BytesIO(f.read())
            self.plot_buffer = buf
            os.remove("plot.png")
        else:
            self.plot_buffer = None

        return result.output

executor = PlotAwareExecutor(
    timeout=10,
    work_dir=temp_dir.name,
)

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

# Global agent instances with updated system messages
initial_agent = ConversableAgent(
    name="initial_agent",
    system_message=f"""
{Initial_Agent_Instructions}""",
    human_input_mode="NEVER",
    llm_config=initial_config
)

review_agent = ConversableAgent(
    name="review_agent",
    system_message=f"""{Review_Agent_Instructions}""",
    human_input_mode="NEVER",
    llm_config=review_config
)

typo_agent = ConversableAgent(
    name="typo_agent",
    system_message=f"""You are the typo and code correction agent. Your task is to:
1. Fix any typos or grammatical errors
2. Correct any code issues
3. Ensure proper formatting
4. Maintain the original meaning while improving clarity
5. Verify plots are saved to disk (not using show())
6. PRESERVE all code blocks exactly as they are unless there are actual errors
7. If no changes are needed, keep the original code blocks unchanged

{Typo_Agent_Instructions}""",
    human_input_mode="NEVER",
    llm_config=typo_config
)

formatting_agent = ConversableAgent(
    name="formatting_agent",
    system_message="""
You are a code formatting assistant.
Your job is to prepare Python code for automated execution.

1. If the message contains code blocks (text between ```python and ```):
   - Extract the code from the blocks
   - If the code uses matplotlib:
     - Remove all 'plt.show()' calls
     - Add these lines at the end:
       fig = plt.gcf()
       fig.savefig("plot.png", dpi=300, bbox_inches='tight')
       plt.close('all')
   - Wrap the modified code back in ```python and ``` blocks
   - Return the complete message with:
     a) The original explanation text
     b) The modified code blocks
     c) A note about code execution: "💡 Note: This answer contains code. If you want to execute it, type 'execute!' in the chat."

2. If the message does not contain code blocks:
   - Format the text in a clear, professional manner
   - Use proper markdown formatting
   - Maintain readability and clarity

3. Always return the complete message with any code blocks properly formatted and modified as needed.

Example input:
```
Here's some code:
```python
import matplotlib.pyplot as plt
plt.plot([1,2,3])
plt.show()
```
```

Example output:
```
Here's some code:
```python
import matplotlib.pyplot as plt
plt.plot([1,2,3])
fig = plt.gcf()
fig.savefig("plot.png", dpi=300, bbox_inches='tight')
plt.close('all')
```

💡 Note: This answer contains code. If you want to execute it, type 'execute!' in the chat.
```
""",
    human_input_mode="NEVER",
    llm_config=formatting_config
)

code_executor = ConversableAgent(
    name="code_executor",
    system_message="""You are the code execution agent. Your task is to:
1. Extract any code blocks from the message (text between ```python and ```)
2. Execute the extracted code and report the results
3. If the code execution fails, provide error details
4. If no code blocks are found, respond with "No code blocks found to execute"
5. For matplotlib plots, ensure they are saved to disk instead of using .show()
6. ALWAYS check for code blocks in the message
7. If code blocks are found, execute them and report the results

Example response format:
```
Code Execution Results:
exitcode: 0 (execution succeeded)
Code output: [output here]
```

If there are errors:
```
Code Execution Results:
exitcode: 1 (execution failed)
Error: [error details here]
```""",
    human_input_mode="NEVER",
    llm_config=code_execution_config,
    code_execution_config={"executor": executor},
    max_consecutive_auto_reply=50
)

def call_ai(context, user_input):
    if mode_is_fast:
        messages = build_messages(context, user_input, Classy_instuctions)
        response = st.session_state.llm.invoke(messages)
        return Response(content=response.content)
    else:
        # New Swarm Workflow for detailed mode
        st.markdown("Thinking (Swarm Mode)... ")

        # Format the conversation history for context
        conversation_history = format_memory_messages(st.session_state.memory.messages)

        # 1. Initial Agent generates the draft
        st.markdown("Generating initial draft...")
        chat_result_1 = initial_agent.initiate_chat(
            recipient=initial_agent,
            message=f"Conversation history:\n{conversation_history}\n\nContext from documents: {context}\n\nUser question: {user_input}",
            max_turns=1,
            summary_method="last_msg"
        )
        draft_answer = chat_result_1.summary
        if st.session_state.debug: st.text(f"Initial Draft:\n{draft_answer}")

        # 2. Review Agent critiques the draft
        st.markdown("Reviewing draft...")
        chat_result_2 = review_agent.initiate_chat(
            recipient=review_agent,
            message=f"Conversation history:\n{conversation_history}\n\nPlease review this draft answer:\n{draft_answer}",
            max_turns=1,
            summary_method="last_msg"
        )
        review_feedback = chat_result_2.summary
        if st.session_state.debug: st.text(f"Review Feedback:\n{review_feedback}")

        # 3. Typo Agent corrects the draft
        st.markdown("Checking for typos...")
        chat_result_3 = typo_agent.initiate_chat(
            recipient=typo_agent,
            message=f"Original draft: {draft_answer}\n\nReview feedback: {review_feedback}",
            max_turns=1,
            summary_method="last_msg"
        )
        typo_corrected_answer = chat_result_3.summary
        if st.session_state.debug: st.text(f"Typo-Corrected Answer:\n{typo_corrected_answer}")

        # 4. Formatting Agent formats the final answer
        st.markdown("Formatting final answer...")
        chat_result_4 = formatting_agent.initiate_chat(
            recipient=formatting_agent,
            message=f"""Please format this answer while preserving any code blocks:

{typo_corrected_answer}

Remember to:
1. Keep all code blocks exactly as they are
2. Add proper matplotlib save commands if needed
3. Format the text professionally""",
            max_turns=1,
            summary_method="last_msg"
        )
        formatted_answer = chat_result_4.summary
        if st.session_state.debug: st.text(f"Formatted Answer:\n{formatted_answer}")

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
                execution_output = executor.execute_code(last_assistant_message)
                
                # Display execution results
                st.markdown("### Execution Results")
                
                # Display the plot if available
                if executor.plot_buffer:
                    st.markdown("### Plot Output")
                    st.image(executor.plot_buffer, use_column_width=True)
                else:
                    st.warning("No plot was generated.")
                
                formatted_execution_results = format_execution_results(execution_output, executor.plot_buffer)
                st.markdown(formatted_execution_results)

                # Check for errors and iterate if needed
                max_iterations = 3  # Maximum number of iterations to prevent infinite loops
                current_iteration = 0
                has_errors = "Error in Class" in execution_output

                while has_errors and current_iteration < max_iterations:
                    current_iteration += 1
                    st.markdown(f"Fixing errors (attempt {current_iteration}/{max_iterations})...")

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

                    # Get corrected version
                    chat_result_3 = typo_agent.initiate_chat(
                        recipient=typo_agent,
                        message=f"""Original answer: {last_assistant_message}

Review feedback with error fixes: {review_feedback}

IMPORTANT: Only fix actual errors in the code blocks. Preserve all working code exactly as it is.""",
                        max_turns=1,
                        summary_method="last_msg"
                    )
                    typo_corrected_answer = chat_result_3.summary

                    # Format the corrected answer
                    chat_result_4 = formatting_agent.initiate_chat(
                        recipient=formatting_agent,
                        message=f"""Please format this corrected answer while preserving all code blocks:

{typo_corrected_answer}

Remember to:
1. Keep all code blocks exactly as they are
2. If the code uses matplotlib, add the save commands at the end
3. Format the text professionally""",
                        max_turns=1,
                        summary_method="last_msg"
                    )
                    formatted_answer = chat_result_4.summary

                    # Execute the corrected code
                    st.info("🚀 Executing corrected code...")
                    execution_output = executor.execute_code(formatted_answer)
                    
                    # Display new execution results
                    st.markdown(f"### Execution Results (Attempt {current_iteration})")
                    
                    # Display the plot if available
                    if executor.plot_buffer:
                        st.markdown("### Plot Output")
                        st.image(executor.plot_buffer, use_column_width=True)
                    else:
                        st.warning("No plot was generated.")
                    
                    formatted_execution_results = format_execution_results(execution_output, executor.plot_buffer)
                    st.markdown(formatted_execution_results)

                    # Update last_assistant_message with the formatted answer for next iteration
                    last_assistant_message = formatted_answer
                    has_errors = "Error in Class" in execution_output

                if has_errors:
                    st.markdown("> ⚠️ **Note**: Some errors could not be fixed after multiple attempts. You can request changes by describing them in the chat.")
                    
                    # Create a response object with the execution results
                    response = Response(content=formatted_execution_results)
                else:
                    st.markdown("> ❌ No code found to execute in the previous messages.")
                    response = Response(content="No code found to execute.")
        else:
            response = call_ai(context, user_input)

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
