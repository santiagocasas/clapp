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

        # 3) Capture stdout and stderr while executing
        with self._capture_output() as (out_buf, err_buf):
            try:
                # Execute in a fresh global namespace exposing only plt
                exec_globals = {"__name__": "__main__", "plt": plt}
                exec(cleaned, exec_globals)
            except Exception:
                # Print full traceback into stderr buffer
                import traceback
                traceback.print_exc(file=sys.stderr)

        # 4) Gather text output
        stdout_text = out_buf.getvalue()
        stderr_text = err_buf.getvalue()
        full_output = ""
        if stdout_text:
            full_output += f"STDOUT:\n{stdout_text}"
        if stderr_text:
            full_output += f"\nSTDERR:\n{stderr_text}"

        # 5) Capture the current Matplotlib figure into PNG bytes
        fig = plt.gcf()
        img_buf = None
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            img_buf = buf
        except Exception:
            img_buf = None

        # 6) Clear the figure so next run starts fresh
        plt.clf()

        return full_output, img_buf

    def __del__(self):
        # Cleanup the temporary directory
        if hasattr(self, "_temp_dir"):
            self._temp_dir.cleanup()


# ------------------------------------------
# Example: using PlotAwareExecutor in Streamlit
# ------------------------------------------

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
    output_text, img_buf = executor.execute_code(code)

    st.subheader("Execution Output")
    st.text(output_text or "— no output —")

    if img_buf:
        st.subheader("Generated Plot")
        # Display the PNG directly
        st.image(img_buf.getvalue(), use_column_width=True)
    else:
        st.info("No plot was generated.")
