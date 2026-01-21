import os

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def configure_environment():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf_cache")


def load_local_secrets():
    secrets_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), ".streamlit", "secrets.toml"
    )
    if not os.path.exists(secrets_path):
        return {}
    try:
        with open(secrets_path, "rb") as file:
            secrets = tomllib.load(file)
        return secrets if isinstance(secrets, dict) else {}
    except Exception:
        return {}


def get_local_secret(key: str):
    secrets = load_local_secrets()
    if secrets:
        return secrets.get(key)
    try:
        import streamlit as st

        return st.secrets.get(key)
    except Exception:
        return None


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/"


GPT_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1"]
GEMINI_MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]
BLABLADOR_MODELS = ["alias-fast", "alias-large", "alias-code", "alias-embeddings"]

DEFAULT_MODEL = "alias-large"
DEFAULT_BLABLADOR_BASE_URL = "https://api.helmholtz-blablador.fz-juelich.de/v1/"
DEFAULT_EMBEDDING_PROVIDER = "Blablador (alias-embeddings)"
