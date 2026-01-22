from langchain_core.messages import HumanMessage, SystemMessage


def build_messages(context, question, system, memory_messages):
    system_msg = SystemMessage(content=system)
    human_msg = HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{question}")
    return [system_msg] + memory_messages + [human_msg]


def format_memory_messages(memory_messages):
    formatted = ""
    for msg in memory_messages:
        role = msg.type.capitalize()
        content = msg.content
        formatted += f"{role}: {content}\n\n"
    return formatted.strip()


def retrieve_context(vector_store, question, k=4):
    try:
        import streamlit as st

        from clapp.config import (
            DEFAULT_BLABLADOR_BASE_URL,
            DEFAULT_EMBEDDING_PROVIDER,
            get_local_secret,
        )
        from clapp.llms.providers import build_embeddings

        embedding_provider = st.session_state.get(
            "embedding_provider", DEFAULT_EMBEDDING_PROVIDER
        )
        if embedding_provider == "Blablador (alias-embeddings)":
            base_url = (
                st.session_state.get("blablador_base_url")
                or get_local_secret("BLABLADOR_BASE_URL")
                or DEFAULT_BLABLADOR_BASE_URL
            )
            api_key = st.session_state.get("saved_api_key_blablador") or get_local_secret(
                "BLABLADOR_API_KEY"
            )
            embeddings = build_embeddings(embedding_provider, api_key, base_url)
            embedding = embeddings.embed_query(question)
            docs = vector_store.similarity_search_by_vector(embedding, k=k)
        else:
            docs = vector_store.similarity_search(question, k=k)
    except Exception as exc:
        try:
            import streamlit as st

            st.info(
                "Embeddings are temporarily unavailable. Continuing without retrieval context."
            )
            if st.session_state.get("debug"):
                st.caption(str(exc))
        except Exception:
            pass
        return "", []
    context = "\n\n".join([doc.page_content for doc in docs])
    evidence = []
    for doc in docs:
        metadata = doc.metadata or {}
        evidence.append(
            {
                "source": metadata.get("source", "unknown"),
                "type": metadata.get("type", "text"),
                "content": doc.page_content,
            }
        )
    return context, evidence
