import os
import subprocess
import sys
import tempfile
import time

import requests
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory

from clapp.config import (
    BLABLADOR_MODELS,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_MODEL,
    GEMINI_MODELS,
    GPT_MODELS,
    normalize_base_url,
)
from clapp.llms.providers import build_embeddings
from clapp.rag.ingest import get_all_docs_from_class_data
from clapp.rag import store as rag_store
from clapp.rag.store import FAISS
from clapp.utils.security import get_fernet, load_encrypted_key, save_encrypted_key


class SidebarState:
    def __init__(self, api_key, api_key_gai, options):
        self.api_key = api_key
        self.api_key_gai = api_key_gai
        self.options = options


def blablador_get_models(base_url: str, api_key: str):
    try:
        url = f"{normalize_base_url(base_url)}models"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None, f"Blablador error {response.status_code}: {response.text}"
        try:
            return response.json(), None
        except ValueError:
            return None, "Blablador returned non-JSON response."
    except requests.RequestException as exc:
        return None, f"Request failed: {exc}"
    except Exception as exc:
        return None, f"Blablador request failed: {exc}"


def render_sidebar(base_dir: str) -> SidebarState:
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-title">🔐 API & Assistants</div>',
            unsafe_allow_html=True,
        )
        api_key = st.text_input("1. OpenAI API Key", type="password")
        api_key_gai = st.text_input("1. Gemini API Key", type="password")

        if api_key:
            st.session_state.saved_api_key = api_key
        if api_key_gai:
            st.session_state.saved_api_key_gai = api_key_gai

        st.markdown(
            '<div class="sidebar-title">Blablador (Helmholtz AI)</div>',
            unsafe_allow_html=True,
        )
        if st.session_state.saved_api_key_blablador:
            st.caption("Blablador API key loaded from secrets.toml.")
        else:
            st.warning("No BLABLADOR_API_KEY found in secrets.toml.")
        st.caption(f"Blablador Base URL: {st.session_state.blablador_base_url}")

        if st.button("Test Blablador models"):
            if not st.session_state.saved_api_key_blablador:
                st.error("Please enter your Blablador API key first.")
            else:
                with st.spinner("Testing Blablador models..."):
                    try:
                        models, error = blablador_get_models(
                            st.session_state.blablador_base_url,
                            st.session_state.saved_api_key_blablador,
                        )
                        if error:
                            st.error(error)
                        else:
                            try:
                                st.json(models)
                            except Exception as exc:
                                st.error(f"Failed to render models: {exc}")
                    except Exception as exc:
                        st.error(f"Blablador test failed: {exc}")

        username = st.text_input(
            "2. Username (for loading or saving API key)",
            placeholder="Enter your username",
        )
        user_password = st.text_input(
            "3. Password to encrypt/decrypt API key", type="password"
        )

        username_display = username if username else "anon"
        openai_file = f"{username_display}_encrypted_api_key"
        gemini_file = f"{username_display}_gai_encrypted_api_key"
        openai_file_exists = os.path.exists(openai_file)
        gemini_file_exists = os.path.exists(gemini_file)

        openai_loaded = bool(st.session_state.get("saved_api_key"))
        gemini_loaded = bool(st.session_state.get("saved_api_key_gai"))

        st.markdown(
            f"OpenAI Key: {'✅ Ready' if openai_loaded else '❌ No Keys'} | "
            f"Saved: {'🗄️' if openai_file_exists else '—'}"
        )
        st.markdown(
            f"Gemini Key: {'✅ Ready' if gemini_loaded else '❌ No Keys'} | "
            f"Saved: {'🗄️' if gemini_file_exists else '—'}"
        )

        if (
            (openai_loaded or gemini_loaded)
            and user_password
            and username
            and (not openai_file_exists or not gemini_file_exists)
        ):
            if st.button("💾 Save API Key(s) as encrypted file"):
                fernet = get_fernet(user_password)
                try:
                    if openai_loaded and not openai_file_exists:
                        if st.session_state.saved_api_key:
                            encrypted_key = fernet.encrypt(
                                st.session_state.saved_api_key.encode()
                            )
                            if save_encrypted_key(
                                encrypted_key.decode(), username_display
                            ):
                                st.success("OpenAI API key encrypted and saved! ✅")
                            else:
                                st.warning(
                                    "OpenAI API key encrypted but couldn't save to file! ⚠️"
                                )
                    if gemini_loaded and not gemini_file_exists:
                        if st.session_state.saved_api_key_gai:
                            encrypted_key_gai = fernet.encrypt(
                                st.session_state.saved_api_key_gai.encode()
                            )
                            if save_encrypted_key(
                                encrypted_key_gai.decode(), f"{username_display}_gai"
                            ):
                                st.success("Gemini API key encrypted and saved! ✅")
                            else:
                                st.warning(
                                    "Gemini API key encrypted but couldn't save to file! ⚠️"
                                )
                except Exception as exc:
                    st.error(f"Error saving API key: {str(exc)}")
                st.rerun()

        if st.button("🔐 Load Saved API Key(s)"):
            if not username or not user_password:
                st.error(
                    "Please enter both username and password to load saved API key(s)."
                )
            else:
                fernet = get_fernet(user_password)
                error = False
                try:
                    if openai_file_exists:
                        encrypted_key = load_encrypted_key(username_display)
                        if encrypted_key:
                            decrypted_key = fernet.decrypt(encrypted_key.encode()).decode()
                            st.session_state.saved_api_key = decrypted_key
                            st.success("OpenAI API key loaded from encrypted file! 🔑")
                    if gemini_file_exists:
                        encrypted_key_gai = load_encrypted_key(
                            f"{username_display}_gai"
                        )
                        if encrypted_key_gai:
                            decrypted_key_gai = fernet.decrypt(
                                encrypted_key_gai.encode()
                            ).decode()
                            st.session_state.saved_api_key_gai = decrypted_key_gai
                            st.success("Gemini API key loaded from encrypted file! 🔑")
                except Exception:
                    st.error(
                        "Failed to decrypt API key(s): Please check your username and password."
                    )
                    error = True
                if not error:
                    if (
                        st.session_state.saved_api_key
                        and st.session_state.selected_model in GPT_MODELS
                    ) or (
                        st.session_state.saved_api_key_gai
                        and st.session_state.selected_model in GEMINI_MODELS
                    ):
                        st.session_state.llm_initialized = True
                    st.rerun()

        if openai_file_exists or gemini_file_exists:
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
                except Exception as exc:
                    error_message += f"Error clearing file: {str(exc)}\n"
                for key in [
                    "saved_api_key",
                    "saved_api_key_gai",
                    "encrypted_key",
                    "encrypted_key_gai",
                ]:
                    if key in st.session_state:
                        del st.session_state[key]
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

        if not api_key_gai:
            st.markdown("Get a Gemini API key from https://aistudio.google.com/apikey")

        options = []
        if st.session_state.saved_api_key_blablador:
            options += BLABLADOR_MODELS
        if api_key_gai:
            options += GEMINI_MODELS
        if api_key:
            options += GPT_MODELS

        if options:
            st.markdown("---")

            if st.session_state.selected_model not in options:
                st.session_state.selected_model = (
                    DEFAULT_MODEL if DEFAULT_MODEL in options else options[0]
                )
            st.session_state.selected_model = st.selectbox(
                "4. Choose LLM model",
                options=options,
                index=options.index(st.session_state.selected_model),
            )
        else:
            st.session_state.selected_model = None

        if "previous_model" not in st.session_state:
            st.session_state.previous_model = st.session_state.selected_model
        elif st.session_state.previous_model != st.session_state.selected_model:
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
        elif (
            st.session_state.selected_model in BLABLADOR_MODELS
            and st.session_state.saved_api_key_blablador
        ):
            st.session_state.llm_initialized = True

        if options:
            st.write("### Response Mode")
            st.radio(
                "Response Mode",
                options=["Fast Mode", "Deep Thought Mode"],
                index=0
                if st.session_state.get("mode_is_fast", "Fast Mode") == "Fast Mode"
                else 1,
                horizontal=True,
                key="mode_is_fast",
                label_visibility="collapsed",
            )

            st.markdown("<div style='height: 0.5em'></div>", unsafe_allow_html=True)
            desc_cols = st.columns(2)
            with desc_cols[0]:
                st.caption(
                    "✨ **Fast Mode**: Single agent setup, quick responses with good quality, "
                    "but prone to initial errors."
                )
            with desc_cols[1]:
                st.caption(
                    "🎯 **Deep Thought Mode**: Multi-agent setup, responses take longer, "
                    "more refined, more accurate at first attempt."
                )

        else:
            st.session_state.mode_is_fast = "Fast Mode"

        st.markdown("---")

        st.markdown(
            '<div class="sidebar-title">🧩 RAG data & Embeddings</div>',
            unsafe_allow_html=True,
        )
        if "embedding_provider" not in st.session_state:
            st.session_state.embedding_provider = DEFAULT_EMBEDDING_PROVIDER

        embedding_provider_options = ["HuggingFace (local)"]
        if st.session_state.saved_api_key_blablador:
            embedding_provider_options.append("Blablador (alias-embeddings)")
        if st.session_state.embedding_provider not in embedding_provider_options:
            st.session_state.embedding_provider = embedding_provider_options[0]

        st.session_state.embedding_provider = st.selectbox(
            "Embedding provider",
            options=embedding_provider_options,
            index=embedding_provider_options.index(st.session_state.embedding_provider),
        )

        def generate_and_save_embedding(embedding_provider):
            embeddings = build_embeddings(
                embedding_provider,
                st.session_state.saved_api_key_blablador,
                st.session_state.blablador_base_url,
            )
            all_docs = get_all_docs_from_class_data()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )

            def sanitize(documents):
                for doc in documents:
                    doc.page_content = doc.page_content.encode("utf-8", "ignore").decode(
                        "utf-8"
                    )
                return documents

            splits = text_splitter.split_documents(all_docs)
            splits = sanitize(splits)
            st.session_state.vector_store = FAISS.from_documents(
                splits, embedding=embeddings
            )
            rag_store.save_vector_store(
                st.session_state.vector_store, embedding_provider
            )

        if "vector_store" not in st.session_state:
            st.session_state.vector_store = None

        embedding_status = st.empty()
        index_file = rag_store.index_file_for_provider(
            st.session_state.embedding_provider
        )
        index_exists = os.path.exists(index_file)

        if st.session_state.vector_store:
            st.markdown("✅ Embedding loaded from file")
            if st.button("🔄 Regenerate embedding"):
                embedding_status.info(
                    "🔄 Processing and embedding your RAG data... This might take a moment! ⏳"
                )
                generate_and_save_embedding(st.session_state.embedding_provider)
                embedding_status.empty()
                st.rerun()
        elif index_exists:
            st.markdown(
                "🗂️ Embedding file found on disk, but not loaded. "
                "Please load the embedding to use the agents!"
            )
            with st.spinner("Loading embeddings..."):
                embeddings = build_embeddings(
                    st.session_state.embedding_provider,
                    st.session_state.saved_api_key_blablador,
                    st.session_state.blablador_base_url,
                )
                st.session_state.vector_store = rag_store.load_vector_store(
                    embeddings, st.session_state.embedding_provider
                )
                st.rerun()
            if st.button("🔄 Regenerate embedding"):
                embedding_status.info(
                    "🔄 Processing and embedding your RAG data... This might take a moment! ⏳"
                )
                generate_and_save_embedding(st.session_state.embedding_provider)
                embedding_status.empty()
                st.rerun()
        else:
            st.markdown(
                "⚠️ No embedding found. Please create the embedding to use the agents!"
            )

            if st.button("🚀 Generate embedding"):
                embedding_status.info(
                    "🔄 Processing and embedding your RAG data... This might take a moment! ⏳"
                )
                generate_and_save_embedding(st.session_state.embedding_provider)
                embedding_status.empty()
                st.rerun()

        st.markdown("---")

        has_code_response = any(
            msg["role"] == "assistant" and "```" in msg["content"]
            for msg in st.session_state.get("messages", [])
        )
        execute_disabled = not has_code_response
        st.markdown(
            '<div class="sidebar-title">⚡ Execute Code</div>',
            unsafe_allow_html=True,
        )
        if st.button(
            "Run Last Code Block",
            key="execute_code_btn",
            disabled=execute_disabled,
            help="This button is enabled when the last assistant response contains code.",
        ):
            from clapp.services.orchestrator import run_code_request

            response = run_code_request()
            st.session_state.memory.add_ai_message(response.content)
            st.session_state.messages.append(
                {"role": "assistant", "content": response.content}
            )
            if "```" in response.content:
                st.rerun()
        st.caption(
            'Alternatively, type "plot!" or "execute!" in the chat to run the last code block.'
        )
        st.markdown("---")

        st.markdown(
            '<div class="sidebar-title">🛠️ CLASS Setup</div>',
            unsafe_allow_html=True,
        )
        if st.checkbox("Check CLASS installation status"):
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "from classy import Class; print('CLASS successfully imported!')",
                    ],
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    st.success("✅ CLASS is already installed and ready to use!")
                else:
                    st.error(
                        "❌ The 'classy' module is not installed. "
                        "Please install CLASS using the button below."
                    )
                    if result.stderr:
                        st.code(result.stderr, language="bash")
            except Exception as exc:
                st.error(f"❌ Error checking CLASS installation: {str(exc)}")

        st.text("If not installed, install CLASS to enable code execution and plotting")
        if st.button("🔄 Install CLASS"):
            status_placeholder = st.empty()
            status_placeholder.info("Installing CLASS... This could take a few minutes.")

            try:
                install_script_path = os.path.join(base_dir, "install_classy.sh")
                os.chmod(install_script_path, 0o755)

                process = subprocess.Popen(
                    [install_script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    shell=True,
                    cwd=base_dir,
                )

                current_line_placeholder = st.empty()

                output_text = ""
                if process.stdout:
                    for line in iter(process.stdout.readline, ""):
                        output_text += line
                        if line.strip():
                            current_line_placeholder.info(f"Current: {line.strip()}")

                return_code = process.wait()

                current_line_placeholder.empty()

                if return_code == 0:
                    status_placeholder.success("✅ CLASS installed successfully!")
                else:
                    status_placeholder.error(
                        f"❌ CLASS installation failed with return code: {return_code}"
                    )

                with st.expander("View Full Installation Log", expanded=False):
                    st.code(output_text)

            except Exception as exc:
                status_placeholder.error(f"Installation failed with exception: {str(exc)}")
                st.exception(exc)

        st.text("If CLASS is installed, test the environment")
        if st.button("🧪 Test CLASS"):
            status_placeholder = st.empty()
            status_placeholder.info(
                "Testing CLASS environment... This could take a moment."
            )

            try:
                test_script_path = os.path.join(base_dir, "test_classy.py")

                with tempfile.TemporaryDirectory() as temp_dir:
                    process = subprocess.Popen(
                        [sys.executable, test_script_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        cwd=temp_dir,
                    )

                    current_line_placeholder = st.empty()

                    output_text = ""
                    if process.stdout:
                        for line in iter(process.stdout.readline, ""):
                            output_text += line
                            if line.strip():
                                current_line_placeholder.info(f"Current: {line.strip()}")

                    return_code = process.wait()

                    current_line_placeholder.empty()

                    if return_code == 0:
                        status_placeholder.success("✅ CLASS test completed successfully!")
                    else:
                        status_placeholder.error(
                            f"❌ CLASS test failed with return code: {return_code}"
                        )

                    if (
                        "ModuleNotFoundError" in output_text
                        or "ImportError" in output_text
                    ):
                        st.error(
                            "❌ Python module import error detected. Make sure CLASS is properly installed."
                        )

                    if (
                        "CosmoSevereError" in output_text
                        or "CosmoComputationError" in output_text
                    ):
                        st.error("❌ CLASS computation error detected.")

                    with st.expander("View Full Test Log", expanded=False):
                        st.code(output_text)
                        plot_path = os.path.join(temp_dir, "cmb_temperature_spectrum.png")
                        if os.path.exists(plot_path):
                            st.subheader("Generated CMB Power Spectrum")
                            st.image(plot_path, use_container_width=True)
                        else:
                            st.warning("⚠️ No plot was generated")

            except Exception as exc:
                status_placeholder.error(f"Test failed with exception: {str(exc)}")
                st.exception(exc)

        st.markdown("---")
        st.markdown(
            '<div class="sidebar-title">🔎 Evidence</div>',
            unsafe_allow_html=True,
        )
        st.session_state.show_evidence = st.checkbox("🔍 Show Evidence")
        if st.session_state.show_evidence:
            evidence = st.session_state.get("last_evidence") or []
            if evidence:
                st.caption("Latest answer evidence:")
                for idx, item in enumerate(evidence, start=1):
                    source = item.get("source", "unknown")
                    item_type = item.get("type", "text")
                    label = f"{idx}. {source}"
                    with st.expander(label, expanded=False):
                        st.caption(f"Type: {item_type}")
                        st.text(item.get("content", ""))
            else:
                st.caption("No evidence available yet.")
            error_evidence = st.session_state.get("last_error_evidence") or []
            if error_evidence:
                st.caption("Latest error-fix evidence:")
                for idx, item in enumerate(error_evidence, start=1):
                    source = item.get("source", "unknown")
                    item_type = item.get("type", "text")
                    label = f"{idx}. {source}"
                    with st.expander(label, expanded=False):
                        st.caption(f"Type: {item_type}")
                        st.text(item.get("content", ""))
        st.markdown("---")
        st.markdown(
            '<div class="sidebar-title">🧰 Session Tools</div>',
            unsafe_allow_html=True,
        )
        if st.button("🗑️ Reset Chat"):
            st.session_state.clear()
            st.rerun()

        if st.session_state.last_token_count > 0:
            st.markdown(
                f"🧮 **Last response token usage:** `{st.session_state.last_token_count}` tokens"
            )

        if "generated_plots" in st.session_state and st.session_state.generated_plots:
            with st.expander("📊 Plot Gallery", expanded=False):
                st.write("All plots generated during this session:")
                for plot_path in st.session_state.generated_plots:
                    if os.path.exists(plot_path):
                        st.image(
                            plot_path, width=250, caption=os.path.basename(plot_path)
                        )
                        st.markdown("---")

    return SidebarState(api_key, api_key_gai, options)
