import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def get_all_docs_from_class_data():
    all_docs = []
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "class-data"
    )
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)
        elif filename.endswith((".txt", ".py", ".ini")):
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                all_docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "type": "code" if filename.endswith(".py") else "text",
                        },
                    )
                )
    return all_docs
