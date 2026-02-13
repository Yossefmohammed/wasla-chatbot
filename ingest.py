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

# Paths
CHROMA_DIR = Path("./chroma_db")
DOCS_DIR = Path("docs")

# Supported file types
LOADERS = {
    '.pdf': PyPDFLoader,
    '.txt': TextLoader,
    '.csv': CSVLoader,
    '.docx': UnstructuredWordDocumentLoader,
    '.doc': UnstructuredWordDocumentLoader,
    '.pptx': UnstructuredPowerPointLoader,
    '.ppt': UnstructuredPowerPointLoader,
}

def ingest_documents(force_rebuild: bool = False, chunk_size: int = 1200, chunk_overlap: int = 250):
    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir(parents=True)
        print("📁 'docs' folder created. Add your documents here.")
        return None

    # Clear old DB if forced
    if CHROMA_DIR.exists() and force_rebuild:
        print("⚠️ Clearing old Chroma DB...")
        shutil.rmtree(CHROMA_DIR)

    all_documents = []
    # Load documents
    for ext, loader_cls in LOADERS.items():
        files = list(DOCS_DIR.rglob(f"*{ext}"))
        if not files:
            continue
        for file_path in files:
            try:
                loader = loader_cls(str(file_path))
                docs = loader.load()
                all_documents.extend(docs)
            except Exception as e:
                print(f"❌ Failed to load {file_path.name}: {e}")

    if not all_documents:
        print("⚠️ No supported documents found in 'docs/' folder.")
        return None

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " "]
    )
    texts = splitter.split_documents(all_documents)
    print(f"🔹 Total chunks after splitting: {len(texts)}")

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    print(f"⚡ Using embeddings model on device: {device}")

    # Build vector store
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    vectordb.persist()
    print("✅ Chroma DB built successfully with BGE v1.5.")

    return vectordb


if __name__ == "__main__":
    ingest_documents(force_rebuild=True)
