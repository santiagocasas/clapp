try:
    from langchain_community.chat_message_histories import ChatMessageHistory
except ModuleNotFoundError:  # Streamlit Cloud may omit langchain_community
    from langchain_core.chat_history import InMemoryChatMessageHistory as ChatMessageHistory
