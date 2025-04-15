import streamlit as st
import io
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import subprocess
import tempfile
import re

st.set_page_config(page_title="Plot Executor", page_icon="🧪")
st.title("🧪 Minimal Code Executor with Plot Support")

user_code = st.text_area("Paste your Python code with matplotlib here:", height=200)

if st.button("Run Code"):
    st.subheader("Execution Log")

    # Extract the code block (strip markdown if present)
    match = re.search(r"```(?:python)?\n(.*?)```", user_code, re.DOTALL)
    if match:
        code = match.group(1)
    else:
        code = user_code

    # Remove plt.show()
    code = code.replace("plt.show()", "# removed plt.show()")

    # Append save logic if plotting is detected
    if any(cmd in code for cmd in ["plt.plot", "plt.hist", "plt.scatter", "plt.imshow", "plt.bar"]):
        code += (
            "\nimport matplotlib.pyplot as plt\n"
            "fig = plt.gcf()\n"
            "fig.savefig('plot.png', dpi=300, bbox_inches='tight')\n"
            "plt.close('all')\n"
        )
    print(">>> FINAL CODE TO EXECUTE:\n", code)
    st.code(code, language="python")

    # Save code to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix=".py", delete=False) as f:
        f.write(code)
        temp_code_path = f.name

    # Run code
    result = subprocess.run(["python", temp_code_path], capture_output=True, text=True)
    os.remove(temp_code_path)

    # Show output
    if result.stdout:
        st.text("STDOUT:")
        st.code(result.stdout)
    if result.stderr:
        st.text("STDERR:")
        st.code(result.stderr)

    # Show plot if generated
    if os.path.exists("plot.png"):
        st.image("plot.png")
        os.remove("plot.png")
    else:
        st.warning("No plot was generated.")
