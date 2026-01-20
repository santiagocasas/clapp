import json
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

DEFAULT_EXTENSIONS = [".txt", ".py", ".ini", ".rst"]
CONFIG_PATH = Path(__file__).with_name("ingest_formats.json")


def load_ingest_extensions(config_path: Path = CONFIG_PATH) -> list[str]:
    try:
        with config_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        extensions = data.get("extensions") if isinstance(data, dict) else None
        if isinstance(extensions, list) and extensions:
            normalized = []
            for ext in extensions:
                if not isinstance(ext, str):
                    continue
                ext = ext.strip().lower()
                if not ext:
                    continue
                if not ext.startswith("."):
                    ext = f".{ext}"
                normalized.append(ext)
            if normalized:
                return normalized
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return DEFAULT_EXTENSIONS


def get_all_docs_from_class_data():
    all_docs = []
    extensions = set(load_ingest_extensions())
    text_extensions = {ext for ext in extensions if ext != ".pdf"}
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "class-data"
    )
    matched_files = 0
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf" and ".pdf" in extensions:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)
            matched_files += 1
        elif suffix in text_extensions:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                all_docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "type": "code" if suffix == ".py" else "text",
                        },
                    )
                )
            matched_files += 1
    if matched_files == 0:
        print(
            "Warning: no class-data files matched configured extensions: "
            f"{sorted(extensions)}"
        )
    return all_docs
