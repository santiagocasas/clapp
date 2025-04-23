import streamlit as st
import openai
import os
import io
import re
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from autogen import ConversableAgent, LLMConfig
from autogen.coding import LocalCommandLineCodeExecutor
from autogen.coding import CodeBlock

# --- Custom executor that runs and captures plots ---
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

        print(">>> FINAL EXECUTED CODE:\n", code)

        # Wrap as CodeBlock
        code_block = CodeBlock(language="python", code=code)
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

# --- Streamlit UI ---
st.set_page_config(page_title="LLM Code + Plot Executor", page_icon="🧠")
st.title("🧠 LLM-Powered Code Generator + Formatter + Plot Executor")

openai_api_key = st.text_input("🔑 OpenAI API Key", type="password")
user_prompt = st.text_area("💬 What should the code do?", placeholder="e.g. Plot a sine wave using matplotlib", height=100)
run_button = st.button("Generate, Format, and Run Code")

if openai_api_key and run_button:
    # --- Agents ---
    llm_config = LLMConfig(
        api_type="openai",
        model="gpt-4",
        api_key=openai_api_key,
        temperature=0.4,
    )

    # Main code writer agent
    code_writer_agent = ConversableAgent(
        name="code_writer",
        system_message="You are a helpful coding assistant. When asked, respond with complete Python code in a single block using triple backticks. Use matplotlib for plotting.",
        llm_config=llm_config,
        human_input_mode="NEVER"
    )

    # Formatter agent
    formatter_agent = ConversableAgent(
        name="formatter",
        system_message="""
You are a code formatting assistant.
Your job is to prepare Python code for automated execution.
If the code uses matplotlib, do the following:
1. Remove all 'plt.show()' calls.
2. At the end, insert:

    fig = plt.gcf()
    fig.savefig("plot.png", dpi=300, bbox_inches='tight')
    plt.close('all')

Only return the cleaned Python code inside triple backticks.
""",
        llm_config=llm_config,
        human_input_mode="NEVER"
    )

    executor = PlotAwareExecutor(timeout=10)

    # Step 1: Generate code
    st.info("🤖 Generating code...")
    code_response = code_writer_agent.initiate_chat(
        recipient=code_writer_agent,
        message=user_prompt,
        max_turns=1,
        summary_method="last_msg"
    )
    raw_code = code_response.summary.strip()

    # Step 2: Format code
    st.info("🧼 Formatting code for execution...")
    format_response = formatter_agent.initiate_chat(
        recipient=formatter_agent,
        message=f"Please format this code:\n\n{raw_code}",
        max_turns=1,
        summary_method="last_msg"
    )
    cleaned_code = format_response.summary.strip()

    # Step 3: Execute code
    st.info("🚀 Executing cleaned code...")
    execution_output = executor.execute_code(cleaned_code)

    # Step 4: Display results
    st.subheader("🧾 Original Generated Code")
    st.markdown(raw_code)

    st.subheader("🧼 Cleaned Code")
    st.markdown(cleaned_code)

    st.subheader("📄 Execution Output")
    st.code(execution_output, language="text")

    if executor.plot_buffer:
        st.subheader("📈 Plot Output")
        st.image(executor.plot_buffer)
    else:
        st.warning("No plot was generated.")
