from cryptography.fernet import Fernet


def save_encrypted_key(encrypted_key: str, username: str) -> bool:
    if not username:
        username = "anon"
    try:
        filename = f"{username}_encrypted_api_key" if username else ".encrypted_api_key"
        with open(filename, "w", encoding="utf-8") as file:
            file.write(encrypted_key)
        return True
    except Exception:
        return False


def load_encrypted_key(username: str):
    if not username:
        username = "anon"
    try:
        filename = f"{username}_encrypted_api_key" if username else ".encrypted_api_key"
        with open(filename, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return None


def get_fernet(user_password: str) -> Fernet:
    key = user_password.ljust(32)[:32].encode()
    return Fernet(__import__("base64").urlsafe_b64encode(key))
