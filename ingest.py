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

# Embedding selection based on device
def get_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        model_name = "sentence-transformers/all-MiniLM-L6-v2"  # lightweight CPU model
    else:
        model_name = "BAAI/bge-base-en-v1.5"  # GPU-optimized
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    print(f"⚡ Using embeddings: {model_name} on device: {device}")
    return embeddings

def ingest_documents(force_rebuild: bool = False, chunk_size: int = 1200, chunk_overlap: int = 250):
    # Ensure docs folder exists
    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir(parents=True)
        print("📁 'docs' folder created. Add your documents here and rerun the script.")
        return None

    # Clear old DB if needed
    if CHROMA_DIR.exists() and force_rebuild:
        print("⚠️ Clearing old Chroma DB...")
        shutil.rmtree(CHROMA_DIR)

    all_documents = []
    # Load documents with progress
    for ext, loader_cls in LOADERS.items():
        files = list(DOCS_DIR.rglob(f"*{ext}"))
        if not files:
            continue
        for file_path in tqdm(files, desc=f"Loading {ext} files"):
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

    # Get embeddings
    embeddings = get_embeddings()

    # Build vector store
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    vectordb.persist()
    print("✅ Chroma DB built successfully!")

    return vectordb

if __name__ == "__main__":
    ingest_documents(force_rebuild=True)
