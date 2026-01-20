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


def retrieve_context(vector_store, question):
    docs = vector_store.similarity_search(question, k=4)
    return "\n\n".join([doc.page_content for doc in docs])
