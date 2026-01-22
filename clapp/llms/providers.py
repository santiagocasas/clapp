from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from clapp.config import (
    BLABLADOR_MODELS,
    GEMINI_MODELS,
    GPT_MODELS,
    DEFAULT_MODEL,
    get_openai_base_url,
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
    if model in GEMINI_MODELS:
        return ChatGoogleGenerativeAI(
            model=model,
            callbacks=callbacks,
            api_key=api_key_gai,
            temperature=temperature,
            convert_system_message_to_human=True,
        )
    if model in GPT_MODELS:
        openai_base_url = get_openai_base_url()
        return ChatOpenAI(
            model=model,
            streaming=streaming,
            callbacks=callbacks,
            api_key=api_key,
            base_url=openai_base_url,
            temperature=temperature,
        )
    active_blablador_models = set(blablador_models or BLABLADOR_MODELS)
    if blablador_api_key and (
        model in active_blablador_models or model not in GEMINI_MODELS + GPT_MODELS
    ):
        return ChatOpenAI(
            model=model,
            streaming=streaming,
            callbacks=callbacks,
            api_key=blablador_api_key,
            base_url=normalize_base_url(blablador_base_url),
            temperature=temperature,
        )
    return ChatOpenAI(
        model=model,
        streaming=streaming,
        callbacks=callbacks,
        api_key=api_key,
        temperature=temperature,
    )


def build_embeddings(embedding_provider, blablador_api_key, blablador_base_url):
    if embedding_provider == "Blablador (alias-embeddings)":
        return OpenAIEmbeddings(
            model="alias-embeddings",
            api_key=blablador_api_key,
            base_url=normalize_base_url(blablador_base_url),
        )
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
