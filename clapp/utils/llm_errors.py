def format_llm_error(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    if "connection refused" in lowered or "dial tcp" in lowered:
        return "Model appears unavailable right now. Please try another model."
    if "http 5" in lowered or "internalservererror" in lowered:
        return "Model returned a server error. Please try another model."
    if "timeout" in lowered or "timed out" in lowered:
        return "Model request timed out. Please try another model."
    if "no available clients" in lowered or "unsupported model" in lowered:
        return "Selected model is temporarily unavailable. Please try another model."
    return f"Model request failed: {message}"


def is_model_temporarily_unavailable(exc: Exception) -> bool:
    message = str(exc)
    lowered = message.lower()
    if "no available clients" in lowered:
        return True
    if "unsupported model" in lowered and "no available clients" in lowered:
        return True
    if "model" in lowered and "unavailable" in lowered:
        return True
    return False
