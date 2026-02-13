import os
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader,
    UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
import shutil
from tqdm import tqdm

# Use the same path as in app.py
CHROMA_DIR = Path("./chroma_db")
DOCS_DIR = Path("docs")

def ingest_documents(force_rebuild=False):
    if not DOCS_DIR.exists():
        raise FileNotFoundError("❌ docs folder not found")

    # Load all supported document types
    loaders = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.csv': CSVLoader,
        '.docx': UnstructuredWordDocumentLoader,
        '.pptx': UnstructuredPowerPointLoader,
    }

    all_documents = []
    for ext, loader_cls in loaders.items():
        for file_path in DOCS_DIR.rglob(f"*{ext}"):
            print(f"📄 Loading: {file_path.name}")
            loader = loader_cls(str(file_path))
            docs = loader.load()
            all_documents.extend(docs)

    if not all_documents:
        raise ValueError("❌ No supported documents found in docs/")

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
        separators=["\n\n", "\n", ".", "!", "?", ",", " "]
    )
    texts = splitter.split_documents(all_documents)
    print(f"🔹 Total chunks: {len(texts)}")

    # Use the same embedding model as app.py
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    print(f"⚡ Using device: {device}")

    # Clear old DB if forced
    if CHROMA_DIR.exists() and force_rebuild:
        print("⚠️ Clearing old Chroma DB...")
        shutil.rmtree(CHROMA_DIR)

    # Build and persist
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    vectordb.persist()
    print("✅ Chroma DB built successfully with BGE v1.5.")

if __name__ == "__main__":
    ingest_documents(force_rebuild=True)