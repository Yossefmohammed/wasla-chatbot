import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from constant import CHROMA_SETTINGS
import torch
import shutil
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = Path(CHROMA_SETTINGS.persist_directory)


def ingest_documents(force_rebuild=False):
    all_documents = []

    if not DOCS_DIR.exists():
        raise FileNotFoundError("❌ docs folder not found")

    for pdf_file in tqdm(list(DOCS_DIR.rglob("*.pdf")), desc="Loading PDFs"):
        print(f"📄 Loading: {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        documents = loader.load()
        print(f"🔹 {len(documents)} pages loaded")
        all_documents.extend(documents)

    if not all_documents:
        raise ValueError("❌ No PDF files found in docs/")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
        separators=["\n\n", "\n", ".", "!", "?", ",", " "]
    )
    texts = text_splitter.split_documents(all_documents)
    print(f"🔹 Total chunks created: {len(texts)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    print(f"⚡ Using device: {device}")

    if CHROMA_DIR.exists() and force_rebuild:
        print("⚠️ Clearing old Chroma DB...")
        for item in CHROMA_DIR.glob("*"):
            if item.is_file():
                item.unlink()
            else:
                shutil.rmtree(item)

    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    vectordb.persist()
    print("✅ Chroma DB built successfully with BGE v1.5.")


if __name__ == "__main__":
    ingest_documents(force_rebuild=True)
