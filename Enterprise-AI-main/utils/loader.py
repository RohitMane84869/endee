import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

def load_documents(data_dir: str = "data") -> List[Document]:
    docs: List[Document] = []

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        return docs

    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(file_name)[1].lower()

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext in [".docx", ".doc"]:
                loader = Docx2txtLoader(file_path)
            elif ext in [".txt", ".md"]:
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                continue

            file_docs = loader.load()
            for d in file_docs:
                d.metadata["source_file"] = file_name

            docs.extend(file_docs)

        except Exception as e:
            print(f"Error loading {file_name}: {e}")

    return docs

