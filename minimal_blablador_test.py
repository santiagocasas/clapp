import os
import requests
import tomllib

def load_secrets():
    base_dir = os.path.dirname(__file__)
    secrets_paths = [
        os.path.join(base_dir, ".streamlit", "secrets.toml"),
        os.path.join(base_dir, "secrets.toml"),
    ]
    for secrets_path in secrets_paths:
        if not os.path.exists(secrets_path):
            continue
        try:
            with open(secrets_path, "rb") as file:
                secrets = tomllib.load(file)
            return secrets if isinstance(secrets, dict) else {}
        except Exception:
            continue
    return {}


def load_blablador_api_key():
    api_key = os.getenv("BLABLADOR_API_KEY")
    if api_key:
        return api_key
    secrets = load_secrets()
    return secrets.get("BLABLADOR_API_KEY")


def load_blablador_base_url():
    base_url = os.getenv("BLABLADOR_BASE_URL")
    if base_url:
        return base_url
    secrets = load_secrets()
    return secrets.get("BLABLADOR_BASE_URL")

api_key = load_blablador_api_key()

# Set your Blablador API key
if api_key:
    os.environ["BLABLADOR_API_KEY"] = api_key

# Configuration for Blablador API
config = {
    "api_key": os.getenv("BLABLADOR_API_KEY"),
    "base_url": load_blablador_base_url()
    or "https://api.helmholtz-blablador.fz-juelich.de/v1/",
    "model": "alias-fast",
}

def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/"

def test_api_key():
    if not config["api_key"]:
        return {"error": "Missing BLABLADOR_API_KEY"}
    headers = {"Authorization": f"Bearer {config['api_key']}"}
    response = requests.get(f"{normalize_base_url(config['base_url'])}models", headers=headers, timeout=10)
    return response.json()

def format_models(models_response):
    model_ids = extract_model_ids(models_response)
    if not model_ids:
        return ["(No models found)"]
    return [f"{index}. {model_id}" for index, model_id in enumerate(model_ids, start=1)]


def extract_model_ids(models_response):
    if not isinstance(models_response, dict):
        return []
    items = models_response.get("data", [])
    model_ids = []
    for item in items:
        model_id = item.get("id")
        if model_id:
            model_ids.append(model_id)
    return model_ids


def choose_model(model_ids):
    if not model_ids:
        return config["model"]
    fallback_model = config["model"] if config["model"] in model_ids else model_ids[0]
    while True:
        choice = input(
            f"Choose a model [1-{len(model_ids)}] (default {fallback_model}): "
        ).strip()
        if not choice:
            return fallback_model
        if choice.isdigit():
            selected_index = int(choice)
            if 1 <= selected_index <= len(model_ids):
                return model_ids[selected_index - 1]
        if choice in model_ids:
            return choice
        print("Please enter a model number or id from the list.")

def call_blablador_chat(messages):
    if not config["api_key"]:
        return {"error": "Missing BLABLADOR_API_KEY"}
    url = f"{normalize_base_url(config['base_url'])}chat/completions"
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config["model"],
        "messages": messages,
        "temperature": 0.2,
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
    except requests.RequestException as exc:
        return {"error": f"Request failed: {exc}"}
    if response.status_code != 200:
        return {"error": f"HTTP {response.status_code}: {response.text}"}
    try:
        return response.json()
    except ValueError:
        return {"error": "Non-JSON response from server."}

def run_chat():
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    print(f"Using model: {config['model']}")
    print("Type a message. Use 'exit' to quit.")
    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue
        messages.append({"role": "user", "content": user_text})
        result = call_blablador_chat(messages)
        if "error" in result:
            print("Error:", result["error"])
            continue
        assistant_text = None
        choices = result.get("choices") if isinstance(result, dict) else None
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict):
                    assistant_text = message.get("content")
        if not assistant_text:
            print("Error: Unexpected response format:", result)
            continue
        print("Assistant:", assistant_text)
        messages.append({"role": "assistant", "content": assistant_text})

# Test the API key and interact with the assistant
if __name__ == "__main__":
    api_response = test_api_key()
    print("API Response:", api_response)
    print("\nAvailable models:")
    for line in format_models(api_response):
        print(line)

    model_ids = extract_model_ids(api_response)
    if model_ids:
        config["model"] = choose_model(model_ids)
        print(f"Selected model: {config['model']}")
    else:
        print(f"No models returned. Using default: {config['model']}")

    # Simple chat loop
    run_chat()
