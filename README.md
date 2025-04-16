# CLAPP - CLASS Code Assistant Application

CLAPP is a Streamlit application that acts as an AI assistant specialized in the CLASS cosmology code. It uses LangChain and OpenAI models, leveraging Retrieval-Augmented Generation (RAG) with CLASS documentation and code examples to provide informed responses.

## Features

*   **Conversational AI:** Interact with an AI assistant knowledgeable about CLASS.
*   **RAG Integration:** Retrieves relevant information from CLASS documentation and code (`./class-data/`) to answer questions accurately.
*   **Code Execution:** Capable of executing Python code snippets provided in the chat, including generating and displaying plots.
*   **Multiple Modes:** Offers a "Fast Mode" for quick responses and a "Swarm Mode" for more detailed, multi-agent reviewed answers.
*   **Secure API Key Handling:** Encrypts and saves your OpenAI API key locally.
*   **Model Selection:** Choose between different OpenAI models (e.g., GPT-4o, GPT-4o-mini).

## Setup and Installation

This project uses `uv` for dependency management.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd clapp
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    uv venv
    uv sync
    ```
    This command reads the `uv.lock` file (or `pyproject.toml` if `uv.lock` doesn't exist) and installs the exact dependencies.

3.  **API Key:**
    *   You will need an OpenAI API key.
    *   The application will prompt you for the key and a password to encrypt and store it locally in a `.encrypted_api_key` file. Alternatively, you can set the `OPENAI_API_KEY` environment variable.

4.  **CLASS Data:**
    *   Ensure the `class-data` directory contains the necessary CLASS documentation, code files (.py, .ini, .txt), and potentially PDF documents for the RAG system. The application expects this directory to be present in the root folder.

5.  **Keys and Prompts:**
    *   Ensure the `keys-IDs.json` file exists (even if empty or containing placeholders if not using specific assistant IDs).
    *   Ensure the `prompts/` directory exists and contains the necessary instruction files (`class_instructions.txt`, `review_instructions.txt`, etc.).

## Usage

1.  **Activate the virtual environment:**
    ```bash
    source .venv/bin/activate
    ```
    *(On Windows, use `.venv\Scripts\activate`)*

2.  **Run the Streamlit application:**
    ```bash
    streamlit run CLAPP.py
    ```

3.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

4.  Enter your OpenAI API key and a password when prompted in the sidebar to initialize the application.

## Project Structure

*   `CLAPP.py`: The main Streamlit application script.
*   `pyproject.toml`: Project metadata and dependencies for `uv`.
*   `uv.lock`: Pinned versions of all dependencies.
*   `requirements.txt`: (Potentially outdated) List of dependencies. `uv sync` uses `pyproject.toml` or `uv.lock`.
*   `README.md`: This file.
*   `class-data/`: Directory containing data for the RAG system (CLASS code, docs, etc.).
*   `prompts/`: Directory containing system prompts for the AI agents.
*   `images/`: Contains images used in the app interface.
*   `.encrypted_api_key`: Stores the encrypted OpenAI API key (generated on first run).
*   `keys-IDs.json`: Configuration file (potentially for API keys or assistant IDs, structure depends on specific use).
