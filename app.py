import streamlit as st
import os
import gc
import time
from dotenv import load_dotenv
from typing import List, Tuple
from functools import wraps

# ===============================
# ENV & INIT
# ===============================
load_dotenv()
gc.enable()
gc.collect()

# ===============================
# IMPORTS
# ===============================
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
from dataclasses import dataclass

# ===============================
# SETTINGS
# ===============================
@dataclass
class CHROMA_SETTINGS:
    persist_directory: str = "./chroma_db"

DOCS_DIR = "docs"

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(
    page_title="Wasla Solutions - AI Strategy Consultant",
    page_icon="🚀",
    layout="wide"
)

# ===============================
# HELPER FUNCTIONS
# ===============================
def retry(max_retries=4, delay=2):
    """Retry decorator with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == max_retries - 1:
                        raise e
                    time.sleep(delay * (2 ** i))
        return wrapper
    return decorator

# ===============================
# VECTOR STORE LOADING
# ===============================
@st.cache_resource(ttl=3600)
def load_vectorstore(force_rebuild=False):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    persist_dir = CHROMA_SETTINGS.persist_directory
    os.makedirs(persist_dir, exist_ok=True)

    try:
        db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        if db._collection.count() == 0 or force_rebuild:
            db = load_documents_to_vectorstore(embeddings, persist_dir)
    except Exception:
        db = load_documents_to_vectorstore(embeddings, persist_dir)

    return db


def load_documents_to_vectorstore(embeddings, persist_dir):
    os.makedirs(DOCS_DIR, exist_ok=True)
    docs = []

    loaders = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.csv': CSVLoader,
        '.docx': UnstructuredWordDocumentLoader,
        '.pptx': UnstructuredPowerPointLoader,
    }

    for file in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, file)
        ext = os.path.splitext(file)[1].lower()
        if ext in loaders:
            loader = loaders[ext](path)
            docs.extend(loader.load())

    if not docs:
        # fallback sample
        sample_text = "Wasla Solutions specializes in AI strategy, digital transformation, and advisory services."
        sample_path = os.path.join(DOCS_DIR, "sample.txt")
        with open(sample_path, "w") as f:
            f.write(sample_text)
        docs = TextLoader(sample_path).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
        separators=["\n\n", "\n", ".", "!", "?", ",", " "]
    )
    docs = splitter.split_documents(docs)

    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
    db.persist()
    return db

# ===============================
# LLM LOADING
# ===============================
@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0.5,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        streaming=True,
        max_tokens=1024
    )

# ===============================
# SMART RAG
# ===============================
def load_qa_chain():
    llm = load_llm()
    db = load_vectorstore()
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 12, "lambda_mult": 0.7}
    )

    class SmartRAG:
        def __init__(self):
            self.history = []
            self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        def check_repetition(self, query) -> Tuple[bool, int]:
            if not self.history:
                return False, 0
            q_emb = self.embed_model.encode(query, convert_to_tensor=True)
            count, repeat = 0, False
            for i, (prev_q, _, prev_emb) in enumerate(self.history[-6:]):
                sim = util.pytorch_cos_sim(q_emb, prev_emb).item()
                threshold = 0.75 + (1 - i/20)
                if sim > threshold:
                    count += 1
                    if sim > 0.78:
                        repeat = True
            return repeat, count

        def generate_prompt(self, query, context):
            return f"""
You are a senior consultant at Wasla Solutions.

STRICT RULES:
- Only use information from CONTEXT.
- If missing, say:
  "I don't have information about that in my knowledge base."
- Maximum 5 sentences. No bullet points. Vary structure. Never hallucinate.

CONTEXT:
{context}

QUESTION:
{query}

Consultant answer:
"""

        @retry()
        def call_llm(self, prompt):
            return llm.invoke(prompt).content

        def __call__(self, query):
            docs = retriever.get_relevant_documents(query)
            if not docs:
                return "I don't have information about that in my knowledge base.", [], "business"

            # Re-rank top 4 docs
            q_emb = self.embed_model.encode(query, convert_to_tensor=True)
            docs = sorted(
                docs,
                key=lambda d: util.pytorch_cos_sim(q_emb, self.embed_model.encode(d.page_content, convert_to_tensor=True)).item(),
                reverse=True
            )[:4]

            context = "\n\n".join([f"[Document {i+1}]\n{d.page_content}" for i, d in enumerate(docs)])
            prompt = self.generate_prompt(query, context)
            answer = self.call_llm(prompt)

            # update history
            self.history.append((query, answer, q_emb))
            if len(self.history) > 30:
                self.history = self.history[-30:]

            return answer, docs, "business"

    return SmartRAG()

# ===============================
# STREAMLIT UI
# ===============================
def main():
    st.title("🚀 Wasla AI Strategy Consultant")

    if "qa" not in st.session_state:
        st.session_state.qa = load_qa_chain()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Ask about Wasla's services...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                answer, sources, _ = st.session_state.qa(prompt)
                st.markdown(answer)
                # show sources
                if sources:
                    st.markdown("**Sources:**")
                    for i, doc in enumerate(sources):
                        st.markdown(f"- Document {i+1}: {doc.metadata.get('source', 'Unknown')}")

        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
