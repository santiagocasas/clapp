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