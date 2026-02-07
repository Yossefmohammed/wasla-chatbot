import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from constant import CHROMA_SETTINGS


BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = Path(CHROMA_SETTINGS.persist_directory)


def ingest_documents():
    all_documents = []

    if not DOCS_DIR.exists():
        raise FileNotFoundError("❌ docs folder not found")

    for pdf_file in DOCS_DIR.rglob("*.pdf"):
        print(f"📄 Loading: {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        documents = loader.load()
        all_documents.extend(documents)

    if not all_documents:
        raise ValueError("❌ No PDF files found in docs/")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    texts = text_splitter.split_documents(all_documents)

    embeddings = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"}   # ✅ Cloud-safe
    )

    Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    print("✅ Chroma DB built successfully.")


if __name__ == "__main__":
    ingest_documents()
