from __future__ import annotations

try:
    from langchain_community.chat_message_histories import (
        ChatMessageHistory as _ChatMessageHistory,
    )
except ModuleNotFoundError:  # Streamlit Cloud may omit langchain_community
    try:
        from langchain_core.chat_history import (
            InMemoryChatMessageHistory as _ChatMessageHistory,
        )
    except Exception:  # pragma: no cover

        class _ChatMessageHistory:  # type: ignore[no-redef]
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "ChatMessageHistory is unavailable. Install langchain_community or langchain_core."
                )


ChatMessageHistory = _ChatMessageHistory

__all__ = ["ChatMessageHistory"]
