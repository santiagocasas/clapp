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
    if not isinstance(models_response, dict) or "data" not in models_response:
        return ["(No models found)"]
    items = models_response.get("data", [])
    labels = []
    for item in items:
        model_id = item.get("id", "unknown-id")
        owner = item.get("owned_by", "unknown-owner")
        labels.append(f"- {model_id} (owner: {owner})")
    return labels or ["(No models found)"]

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
        try:
            assistant_text = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
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

    # Simple chat loop
    run_chat()
