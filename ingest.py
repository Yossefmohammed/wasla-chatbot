import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from constant import CHROMA_SETTINGS
from langchain_community.embeddings import SentenceTransformerEmbeddings


def main():
    all_documents = []

    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(f"Loading: {file}")
                loader = PyPDFLoader(os.path.join(root, file))
                documents = loader.load()
                all_documents.extend(documents)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    texts = text_splitter.split_documents(all_documents)

    embeddings = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"}   # 🔥 FIX CUDA OOM
    )

    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=CHROMA_SETTINGS.persist_directory
    )

    print("✅ Ingestion completed and saved successfully.")


if __name__ == "__main__":
    main()
