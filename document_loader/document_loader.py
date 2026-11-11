import os
import tempfile

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader

load_dotenv()

def load_text_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(
            b"Hello, this is a sample text file.\nThis file is used to demonstrate the TextLoader."
        )
        temp_file_path = temp_file.name

        try:
            loader =TextLoader(temp_file_path)
            documents = loader.load()

            for doc in documents:
                print(doc.page_content)

        finally:
            os.remove(temp_file_path)


def pdf_loader(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} document(s) from PDF")
    for i, doc in enumerate(documents):
        print(f"Document {i+1} Content Preview: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")