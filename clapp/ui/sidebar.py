import os
import subprocess
import sys
import tempfile

import requests
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter

from clapp.langchain_compat import ChatMessageHistory

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


def extract_blablador_model_ids(models_payload):
    if not isinstance(models_payload, dict):
        return []
    data = models_payload.get("data", [])
    model_ids = []
    for item in data:
        model_id = item.get("id") if isinstance(item, dict) else None
        if not isinstance(model_id, str) or not model_id:
            continue
        if model_id.startswith("text-"):
            continue
        model_ids.append(model_id)
    return model_ids


def get_live_blablador_models():
    if not st.session_state.saved_api_key_blablador:
        st.session_state.blablador_models = []
        st.session_state.blablador_models_error = None
        st.session_state.blablador_models_key = None
        st.session_state.blablador_models_url = None
        st.session_state.blablador_models_payload = None
        return []

    needs_refresh = (
        "blablador_models" not in st.session_state
        or st.session_state.get("blablador_models_key")
        != st.session_state.saved_api_key_blablador
        or st.session_state.get("blablador_models_url")
        != st.session_state.blablador_base_url
    )
    if needs_refresh:
        models, error = blablador_get_models(
            st.session_state.blablador_base_url,
            st.session_state.saved_api_key_blablador,
        )
        if error:
            st.session_state.blablador_models = []
            st.session_state.blablador_models_error = error
            st.session_state.blablador_models_payload = None
        else:
            st.session_state.blablador_models = extract_blablador_model_ids(models)
            st.session_state.blablador_models_error = None
            st.session_state.blablador_models_payload = models
        st.session_state.blablador_models_key = st.session_state.saved_api_key_blablador
        st.session_state.blablador_models_url = st.session_state.blablador_base_url

    return st.session_state.get("blablador_models", [])


def render_sidebar(base_dir: str) -> SidebarState:
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-title">🔐 API & Assistants</div>',
            unsafe_allow_html=True,
        )
        st.caption("Prefer configuring API keys in .streamlit/secrets.toml.")
        api_key = st.text_input(
            "1. OpenAI API Key (Optional)",
            type="password",
            placeholder="OPENAI_API_KEY",
        )
        api_key_gai = st.text_input(
            "1. Gemini API Key (Optional)",
            type="password",
            placeholder="GEMINI_API_KEY",
        )
        st.caption("Get a Gemini API key from https://aistudio.google.com/apikey")

        if api_key:
            st.session_state.saved_api_key = api_key
        if api_key_gai:
            st.session_state.saved_api_key_gai = api_key_gai

        st.markdown(
            '<div class="sidebar-title">Blablador (Helmholtz AI)</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "Supported by <a href=\"https://sdlaml.pages.jsc.fz-juelich.de/ai/guides/blablador_api_access/\" target=\"_blank\">Blablador</a> and <a href=\"https://www.helmholtz.ai/\" target=\"_blank\">Helmholtz AI</a>.",
            unsafe_allow_html=True,
        )
        blablador_logo = os.path.join(base_dir, "images", "blablador-ng.svg")
        if os.path.exists(blablador_logo):
            st.image(blablador_logo, width=160)
        if st.session_state.saved_api_key_blablador:
            st.caption("Blablador API key loaded from secrets.toml.")
        else:
            st.warning("No BLABLADOR_API_KEY found in secrets.toml.")

        if "show_extended_blablador_models" not in st.session_state:
            st.session_state.show_extended_blablador_models = False
        st.caption("Extended models may include experimental or offline entries.")
        st.session_state.show_extended_blablador_models = st.checkbox(
            "Extended Blablador models",
            value=st.session_state.show_extended_blablador_models,
        )
        if st.session_state.show_extended_blablador_models:
            if not st.session_state.saved_api_key_blablador:
                st.error("Please set BLABLADOR_API_KEY in secrets.toml first.")
            else:
                with st.spinner("Loading Blablador models..."):
                    try:
                        models = get_live_blablador_models()
                        if st.session_state.get("blablador_models_error"):
                            st.error(st.session_state["blablador_models_error"])
                        else:
                            st.caption(f"Loaded {len(models)} models.")
                            with st.expander("View Blablador model response"):
                                st.json(
                                    st.session_state.get("blablador_models_payload", {})
                                )
                    except Exception as exc:
                        st.error(f"Blablador model load failed: {exc}")

        if st.session_state.saved_api_key:
            api_key = st.session_state.saved_api_key
        if st.session_state.saved_api_key_gai:
            api_key_gai = st.session_state.saved_api_key_gai

        options = []
        blablador_models = []
        if st.session_state.saved_api_key_blablador:
            blablador_models = BLABLADOR_MODELS
            if st.session_state.show_extended_blablador_models:
                extended_models = get_live_blablador_models()
                if not extended_models:
                    if st.session_state.get("blablador_models_error"):
                        st.warning(
                            "Could not load live Blablador models. Using aliases instead."
                        )
                else:
                    blablador_models = extended_models
            st.session_state.blablador_models = blablador_models
            options += blablador_models
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

        if st.session_state.selected_model in GEMINI_MODELS and api_key_gai:
            st.session_state.llm_initialized = True
        elif st.session_state.selected_model in GPT_MODELS and api_key:
            st.session_state.llm_initialized = True
        elif (
            st.session_state.selected_model in blablador_models
            and st.session_state.saved_api_key_blablador
        ):
            st.session_state.llm_initialized = True
        else:
            st.session_state.llm_initialized = False

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
            'Alternatively, type "plot!", "execute!", or "run!" in the chat to run the last code block.'
        )
        st.caption(
            "Preflight checks run alias-fast with rules from prompts/preflight_instructions.txt."
        )
        st.session_state.preflight_enabled = st.checkbox(
            "Preflight check before execution",
            value=st.session_state.get("preflight_enabled", True),
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
                        "Install it with `pip install classy`."
                    )
                    if result.stderr:
                        st.code(result.stderr, language="bash")
            except Exception as exc:
                st.error(f"❌ Error checking CLASS installation: {str(exc)}")

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
