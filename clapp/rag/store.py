import os

from langchain_community.vectorstores import FAISS

__all__ = [
    "FAISS",
    "ensure_index_root",
    "index_path_for_provider",
    "index_file_for_provider",
    "load_vector_store",
    "save_vector_store",
]


INDEX_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "indexes", "cosmology"
)


def ensure_index_root():
    os.makedirs(INDEX_ROOT, exist_ok=True)


def index_path_for_provider(embedding_provider: str) -> str:
    ensure_index_root()
    if embedding_provider == "Blablador (alias-embeddings)":
        return os.path.join(INDEX_ROOT, "blablador")
    return os.path.join(INDEX_ROOT, "local")


def index_file_for_provider(embedding_provider: str) -> str:
    return os.path.join(index_path_for_provider(embedding_provider), "index.faiss")


def load_vector_store(embeddings, embedding_provider: str):
    folder_path = index_path_for_provider(embedding_provider)
    if not os.path.exists(os.path.join(folder_path, "index.faiss")):
        return None
    return FAISS.load_local(
        folder_path=folder_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


def save_vector_store(vector_store, embedding_provider: str):
    folder_path = index_path_for_provider(embedding_provider)
    vector_store.save_local(folder_path)
