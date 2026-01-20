from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from clapp.config import (
    BLABLADOR_MODELS,
    GEMINI_MODELS,
    GPT_MODELS,
    DEFAULT_MODEL,
    normalize_base_url,
)


def build_llm(
    selected_model,
    api_key,
    api_key_gai,
    blablador_api_key,
    blablador_base_url,
    blablador_models=None,
    callbacks=None,
    streaming=True,
    temperature=0.2,
):
    model = selected_model or DEFAULT_MODEL
    active_blablador_models = set(blablador_models or BLABLADOR_MODELS)
    if model in active_blablador_models:
        return ChatOpenAI(
            model_name=model,
            streaming=streaming,
            callbacks=callbacks,
            openai_api_key=blablador_api_key,
            base_url=normalize_base_url(blablador_base_url),
            temperature=temperature,
        )
    if model in GEMINI_MODELS:
        return ChatGoogleGenerativeAI(
            model=model,
            callbacks=callbacks,
            google_api_key=api_key_gai,
            temperature=temperature,
            convert_system_message_to_human=True,
        )
    return ChatOpenAI(
        model_name=model,
        streaming=streaming,
        callbacks=callbacks,
        openai_api_key=api_key,
        temperature=temperature,
    )


def build_embeddings(embedding_provider, blablador_api_key, blablador_base_url):
    if embedding_provider == "Blablador (alias-embeddings)":
        return OpenAIEmbeddings(
            model="alias-embeddings",
            openai_api_key=blablador_api_key,
            base_url=normalize_base_url(blablador_base_url),
        )
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
