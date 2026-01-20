import json


def read_keys_from_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)
