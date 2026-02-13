import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
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

    # 🔥 Better chunking for RAG quality
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )

    texts = text_splitter.split_documents(all_documents)

    # 🔥 Production embedding model (BGE v1.5)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},  # cloud-safe
        encode_kwargs={"normalize_embeddings": True}
    )

    # Clear old DB if exists (optional but recommended when changing embedding model)
    if CHROMA_DIR.exists():
        print("⚠️ Clearing old Chroma DB (embedding model changed)...")
        for item in CHROMA_DIR.glob("*"):
            if item.is_file():
                item.unlink()
            else:
                import shutil
                shutil.rmtree(item)

    Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    print("✅ Chroma DB built successfully with BGE v1.5.")


if __name__ == "__main__":
    ingest_documents()
