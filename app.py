import streamlit as st
import os, hashlib, json, time, shutil
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import gc
import torch

# ===============================
# ENV & INIT
# ===============================
load_dotenv()
gc.enable()
gc.collect()

# ===============================
# CONSTANTS / SETTINGS
# ===============================
CHROMA_DIR = Path("./chroma_db")
DOCS_DIR = Path("./docs")

if not os.path.exists("constant.py"):
    with open("constant.py", "w") as f:
        f.write('''from dataclasses import dataclass
@dataclass
class CHROMA_SETTINGS:
    persist_directory: str = "./chroma_db"
''')
from constant import CHROMA_SETTINGS

# ===============================
# THEME
# ===============================
def set_dark_theme():
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #0B1020 0%, #151B2B 100%); color: #EAEAF2;}
    section.main > div { max-width: 1000px; margin: auto; padding: 2rem; }
    textarea { background-color: rgba(17, 24, 39, 0.8) !important; color: #E5E7EB !important;
               border: 1px solid #374151 !important; border-radius: 12px !important; padding: 1rem !important; font-size:16px !important;
               backdrop-filter: blur(10px);}
    button {background: linear-gradient(45deg, #2563EB, #3B82F6) !important; color:white !important;
            border:none !important; border-radius:12px !important; padding:0.6rem 1.2rem !important; font-weight:600 !important;
            transition: transform 0.2s, box-shadow 0.2s !important;}
    button:hover {transform: translateY(-2px); box-shadow:0 8px 20px rgba(37,99,235,0.3) !important;}
    section[data-testid="stSidebar"] button{width:100%; margin:0.2rem 0;}
    footer{visibility:hidden;}
    </style>
    """, unsafe_allow_html=True)

set_dark_theme()

# ===============================
# LANGCHAIN / HF IMPORTS
# ===============================
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader,
    UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
)
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Supported loaders
LOADERS = {
    '.pdf': PyPDFLoader,
    '.txt': TextLoader,
    '.csv': CSVLoader,
    '.docx': UnstructuredWordDocumentLoader,
    '.doc': UnstructuredWordDocumentLoader,
    '.pptx': UnstructuredPowerPointLoader,
    '.ppt': UnstructuredPowerPointLoader,
}

# ===============================
# EMBEDDINGS
# ===============================
def get_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Choose lightweight model for CPU, BGE for GPU
    model_name = "sentence-transformers/all-MiniLM-L6-v2" if device=="cpu" else "BAAI/bge-base-en-v1.5"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    return embeddings, device, model_name

# ===============================
# DOC INGESTION
# ===============================
def build_chroma_db(force_rebuild=False):
    if not DOCS_DIR.exists() or not any(DOCS_DIR.iterdir()):
        st.warning("📂 Please upload documents first in the 'docs' folder.")
        return None

    if CHROMA_DIR.exists() and force_rebuild:
        shutil.rmtree(CHROMA_DIR)

    all_docs = []
    for ext, loader_cls in LOADERS.items():
        for file in DOCS_DIR.rglob(f"*{ext}"):
            try:
                loader = loader_cls(str(file))
                docs = loader.load()
                all_docs.extend(docs)
            except Exception as e:
                st.error(f"Failed to load {file.name}: {e}")

    if not all_docs:
        st.error("❌ No valid documents found.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
    chunks = splitter.split_documents(all_docs)

    embeddings, device, model_name = get_embeddings()

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        client_settings={"telemetry_enabled": False}
    )
    vectordb.persist()
    st.success(f"✅ Chroma DB built successfully with {len(chunks)} chunks using {model_name}!")
    return vectordb

# ===============================
# QA CHAIN
# ===============================
def load_qa_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k":3})
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-60m")
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-60m")
    hf_pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # CPU safe
        max_new_tokens=256,
        temperature=0.7
    )
    llm = HuggingFacePipeline(pipeline=hf_pipe)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# ===============================
# SESSION STATE INIT
# ===============================
def init_session_state():
    defaults = {
        "messages": [], "qa_chain": None, "vectordb": None,
        "current_question": "", "current_answer": "", "current_sources": [],
        "session_id": hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ===============================
# SIDEBAR
# ===============================
def render_sidebar():
    st.sidebar.title("Wasla AI Consultant")
    st.sidebar.markdown("Upload docs or ask questions about Wasla's business.")
    if st.sidebar.button("Build / Rebuild Chroma DB"):
        st.session_state.vectordb = build_chroma_db(force_rebuild=True)
        if st.session_state.vectordb:
            st.session_state.qa_chain = load_qa_chain(st.session_state.vectordb)
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages.clear()

# ===============================
# APP
# ===============================
def main():
    init_session_state()
    render_sidebar()
    st.markdown("<h1 style='text-align:center;'>🤖 Wasla AI Strategy Consultant</h1>", unsafe_allow_html=True)

    # Show chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"]=="assistant" and msg.get("sources"):
                with st.expander("📚 Sources"):
                    for i, s in enumerate(msg["sources"][:3]):
                        st.markdown(f"**{i+1}. {s.metadata.get('source','doc')}**")
                        st.caption(s.page_content[:200]+"...")

    # Chat input
    user_input = st.chat_input("Ask a question...")
    if user_input:
        st.session_state.messages.append({"role":"user","content":user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            if st.session_state.qa_chain is None:
                st.warning("⚠️ Build the Chroma DB first!")
            else:
                try:
                    result = st.session_state.qa_chain({"query": user_input})
                    answer = result.get("result") or result.get("answer") or "No answer."
                    sources = result.get("source_documents", [])
                    st.markdown(answer)
                    st.session_state.messages.append({
                        "role":"assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    st.markdown("<p style='text-align:center;color:#6B7280;'>Powered by Wasla Solutions • AI Strategy Consultant</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
