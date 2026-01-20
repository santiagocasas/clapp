import os
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory

from clapp.config import (
    DEFAULT_BLABLADOR_BASE_URL,
    configure_environment,
    get_local_secret,
)
from clapp.prompts import load_prompts
from clapp.ui.chat import render_chat
from clapp.ui.greeting import maybe_greet
from clapp.ui.sidebar import SidebarState, render_sidebar


configure_environment()
PROMPTS = load_prompts()


st.set_page_config(
    page_title="CLAPP Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto",
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
            color: var(--text-color);
            letter-spacing: 1px;
        }}
        .sidebar-title {{
            font-family: '{font_name}', sans-serif !important;
            font-size: 1.5rem !important;
            font-weight: 600 !important;
            margin-bottom: 0.5rem !important;
            margin-top: 0.5rem !important;
            color: var(--text-color);
            letter-spacing: 0.5px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_global_styles_and_font("Jersey 10")

st.markdown(
    '<div class="main-title">CLAPP: CLASS LLM Agent for Pair Programming</div>',
    unsafe_allow_html=True,
)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    base_dir = os.path.dirname(os.path.dirname(__file__))
    st.image(os.path.join(base_dir, "images", "CLAPP.png"), width=400)


Initial_Agent_Instructions = PROMPTS["initial"]


def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "debug" not in st.session_state:
        st.session_state.debug = False
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "memory" not in st.session_state:
        st.session_state.memory = ChatMessageHistory()
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "last_token_count" not in st.session_state:
        st.session_state.last_token_count = 0
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None
    if "greeted" not in st.session_state:
        st.session_state.greeted = False
    if "debug_messages" not in st.session_state:
        st.session_state.debug_messages = []
    if "saved_api_key" not in st.session_state:
        st.session_state.saved_api_key = None
    if "saved_api_key_gai" not in st.session_state:
        st.session_state.saved_api_key_gai = None
    if "saved_api_key_blablador" not in st.session_state:
        st.session_state.saved_api_key_blablador = get_local_secret("BLABLADOR_API_KEY")
    if "blablador_base_url" not in st.session_state:
        st.session_state.blablador_base_url = (
            get_local_secret("BLABLADOR_BASE_URL") or DEFAULT_BLABLADOR_BASE_URL
        )
    if "agents" not in st.session_state:
        st.session_state.agents = None


init_session()


if st.session_state.debug:
    with st.sidebar.expander("🛠️ Debug Information", expanded=True):
        debug_container = st.container()
        with debug_container:
            st.markdown("### Debug Messages")

            for title, message in st.session_state.debug_messages:
                st.markdown(f"### {title}")
                st.markdown(message)
                st.markdown("---")

    with st.sidebar.expander("🛠️ Context Used"):
        if st.session_state.get("last_context"):
            st.markdown(st.session_state["last_context"])
        else:
            st.markdown("No context retrieved yet.")


try:
    sidebar_state = render_sidebar(base_dir)
except Exception as exc:
    st.error("Sidebar failed to render. See details below.")
    st.exception(exc)
    sidebar_state = SidebarState(None, None, [])

api_key = sidebar_state.api_key
api_key_gai = sidebar_state.api_key_gai
options = sidebar_state.options


try:
    render_chat(options, api_key, api_key_gai, Initial_Agent_Instructions)
except Exception as exc:
    st.error("Chat rendering failed. See details below.")
    st.exception(exc)


try:
    maybe_greet(Initial_Agent_Instructions, api_key, api_key_gai)
except Exception as exc:
    st.error("Greeting failed to render. See details below.")
    st.exception(exc)


def main():
    return None
