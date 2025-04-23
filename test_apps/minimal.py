import os
import re
import sys
import io
import tempfile
import contextlib

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from autogen.coding import LocalCommandLineCodeExecutor


class PlotAwareExecutor(LocalCommandLineCodeExecutor):
    """
    An executor that runs user Python code in-process, captures stdout/stderr,
    and saves any Matplotlib figure to an in-memory PNG for Streamlit.
    """

    def __init__(self, **kwargs):
        # Create a temporary working directory
        temp_dir = tempfile.TemporaryDirectory()
        kwargs["work_dir"] = temp_dir.name
        super().__init__(**kwargs)
        self._temp_dir = temp_dir   # keep it alive

    @contextlib.contextmanager
    def _capture_output(self):
        """
        Context manager to capture both stdout and stderr into StringIO buffers.
        """
        old_out, old_err = sys.stdout, sys.stderr
        buf_out, buf_err = io.StringIO(), io.StringIO()
        sys.stdout, sys.stderr = buf_out, buf_err
        try:
            yield buf_out, buf_err
        finally:
            sys.stdout, sys.stderr = old_out, old_err

    def execute_code(self, code: str):
        """
        Execute Python code, capture stdout/stderr, and capture any Matplotlib plot.
        Returns:
            full_output (str): combined STDOUT/STDERR text
            img_buf     (BytesIO or None): PNG image buffer if a plot was drawn
        """
        # 1) Extract code inside ```python``` fences if present
        match = re.search(r"```(?:python)?\n(.*?)```", code, re.DOTALL)
        cleaned = match.group(1) if match else code

        # 2) Remove any plt.show() calls (we capture the figure ourselves)
        cleaned = re.sub(r"\bplt\.show\(\s*\)", "", cleaned)

         # start a fresh figure
        fig = plt.figure()

        # run user code
        with self._capture_output() as (out_buf, err_buf):
            try:
                exec(cleaned, {"__name__":"__main__", "plt":plt})
            except Exception:
                import traceback; traceback.print_exc(file=sys.stderr)

        # collect text output
        text = out_buf.getvalue() + err_buf.getvalue()

        # if nothing got drawn, fig.get_axes() will be empty
        drawn = bool(fig.get_axes())

        # **don’t clear it here**—return the fig object
        return text, fig if drawn else None

    def __del__(self):
        # Cleanup the temporary directory
        if hasattr(self, "_temp_dir"):
            self._temp_dir.cleanup()


# ------------------------------------------
# Example: using PlotAwareExecutor in Streamlit
# ------------------------------------------

class PlotAwareExecutorOld(LocalCommandLineCodeExecutor):
    """
    A specialized code executor that handles plotting and code execution with plot capture.
    
    This executor extends LocalCommandLineCodeExecutor to:
    1. Handle matplotlib plots by saving them to memory instead of displaying them
    2. Clean code for execution by removing plt.show() calls
    3. Support various image formats (.png, .jpg, .jpeg, .gif, .pdf)
    4. Provide access to generated plots through plot_buffer
    
    Attributes:
        plot_buffer (io.BytesIO): Buffer containing the last generated plot image
        supported_extensions (list): List of supported image file extensions
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the PlotAwareExecutor.
        
        Args:
            **kwargs: Additional arguments passed to LocalCommandLineCodeExecutor
        """
        # Create a temporary directory first
        temp_dir = tempfile.TemporaryDirectory()
        kwargs['work_dir'] = temp_dir.name
        
        # Initialize the parent class with the temporary directory
        super().__init__(**kwargs)
        
        self.plot_buffer = None
        self.supported_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.pdf']
        self._temp_dir = temp_dir  # Store the temp_dir to prevent it from being garbage collected
        

    def _test_environment(self):
        """Test the execution environment by running test_classy.py"""
        try:
            # Get the path to test_classy.py
            test_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_classy.py')
            
            if st.session_state.debug:
                st.session_state.debug_messages.append(("Environment Test", f"Testing with script: {test_script_path}"))
                st.session_state.debug_messages.append(("Environment Test", f"Python executable: {sys.executable}"))

            # Run the test script
            result = subprocess.run(
                [sys.executable, test_script_path],
                capture_output=True,
                text=True,
                cwd=self.work_dir,
                timeout=30
            )

            if st.session_state.debug:
                st.session_state.debug_messages.append(("Environment Test", f"Return code: {result.returncode}"))
                st.session_state.debug_messages.append(("Environment Test", f"Output: {result.stdout}"))
                if result.stderr:
                    st.session_state.debug_messages.append(("Environment Test", f"Errors: {result.stderr}"))

            # Check if the plot was generated
            plot_path = os.path.join(self.work_dir, 'cmb_temperature_spectrum.png')
            if os.path.exists(plot_path):
                if st.session_state.debug:
                    st.session_state.debug_messages.append(("Environment Test", f"Plot generated: {plot_path}"))
                # Load the plot into buffer
                with open(plot_path, "rb") as f:
                    self.plot_buffer = io.BytesIO(f.read())
                os.remove(plot_path)
            else:
                if st.session_state.debug:
                    st.session_state.debug_messages.append(("Environment Test", "No plot was generated"))

        except Exception as e:
            if st.session_state.debug:
                st.session_state.debug_messages.append(("Environment Test", f"Test failed: {str(e)}"))
            st.error(f"Environment test failed: {str(e)}")

    def execute_code(self, code: str):
        """
        Execute Python code and capture any generated plots.
        
        This method:
        1. Extracts code from markdown code blocks if present
        2. Cleans the code for execution (removes plt.show() calls)
        3. Executes the code using the system Python interpreter
        4. Captures any generated plots into memory
        
        Args:
            code (str): Python code to execute, optionally wrapped in markdown code blocks
            
        Returns:
            str: The output from code execution
        """
        # Extract code from triple backticks if needed
        match = re.search(r"```(?:python)?\n(.*?)```", code, re.DOTALL)
        if match:
            code = match.group(1)

        # Clean the code for execution
        cleaned_code = clean_code_for_execution(code)

        if st.session_state.debug:
            st.session_state.debug_messages.append((">>Executed Code>>", cleaned_code))

        # Write the code to a temporary script
        script_path = os.path.join(self.work_dir, "script.py")
        with open(script_path, "w") as f:
            f.write(cleaned_code)

        # Execute the script using the system Python interpreter
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                cwd=self.work_dir,
                timeout=10
            )
            output = result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            output = "Execution timed out after 10 seconds"
        except Exception as e:
            output = f"Error executing script: {str(e)}"

        # Capture plot
        if st.session_state.debug:
            st.session_state.debug_messages.append(("Plot Capture", f"Looking for plot files in {self.work_dir}"))
            st.session_state.debug_messages.append(("Plot Capture", f"Files found: {os.listdir(self.work_dir)}"))
        
        for ext in self.supported_extensions:
            for file in os.listdir(self.work_dir):
                if file.endswith(ext):
                    file_path = os.path.join(self.work_dir, file)
                    if st.session_state.debug:
                        st.session_state.debug_messages.append(("Plot Capture", f"Found plot file: {file_path}"))
                    with open(file_path, "rb") as f:
                        buf = io.BytesIO(f.read())
                        if st.session_state.debug:
                            st.session_state.debug_messages.append(("Plot Capture", f"Buffer size: {buf.getbuffer().nbytes} bytes"))
                    self.plot_buffer = buf
                    os.remove(file_path)
                    break
            if self.plot_buffer is not None:
                break
        else:
            if st.session_state.debug:
                st.session_state.debug_messages.append(("Plot Capture", "No plot files found"))
            self.plot_buffer = None

        return output

    def __del__(self):
        """Clean up temporary directory when the executor is destroyed"""
        if hasattr(self, '_temp_dir'):
            self._temp_dir.cleanup()
executor = PlotAwareExecutor(timeout=10)

st.title("Python Runner with Plot Capture")

code = st.text_area(
    "Enter Python code (you can use `plt.plot(...)` etc.):",
    value="""
import numpy as np
x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x))
print("Plotted a sine wave!")
"""
)

if st.button("Run Code"):
    output_text, fig = executor.execute_code(code)
    st.text(output_text or "— no output —")

    if fig:
        st.subheader("Generated Plot")
        st.pyplot(fig)
        plt.close(fig)  # clean up now that Streamlit has it
    else:
        st.info("No plot was generated.")