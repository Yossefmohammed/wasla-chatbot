import streamlit as st
import os
import csv
import gc
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ===============================
# ENV
# ===============================
load_dotenv()

try:
    if hasattr(st, "secrets") and st.secrets:
        for k, v in st.secrets.items():
            if k not in os.environ:
                os.environ[k] = str(v)
except Exception:
    pass

gc.enable()
gc.collect()

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate

try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None

from constant import CHROMA_SETTINGS

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Wasla Solutions", layout="wide")

# ===============================
# DARK THEME
# ===============================
def set_dark_theme():
    st.markdown("""
    <style>
    .stApp { background-color: #0B1020; color: #EAEAF2; }
    section.main > div { max-width: 900px; margin: auto; }
    textarea { background-color: #111827; color: #E5E7EB; border-radius: 10px; }
    button { background-color: #2563EB !important; color: white !important; border-radius: 10px; width: 100%; }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

set_dark_theme()

# ===============================
# VECTOR STORE
# ===============================
@st.cache_resource
def load_vectorstore():
    embeddings = SentenceTransformerEmbeddings(
        model_name=os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    db = Chroma(
        persist_directory=CHROMA_SETTINGS.persist_directory,
        embedding_function=embeddings
    )

    if db._collection.count() == 0:
        folder_path = "docs"
        docs = []

        for file in os.listdir(folder_path):
            if file.lower().endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(folder_path, file))
                docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=200
        )
        docs = splitter.split_documents(docs)

        db = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=CHROMA_SETTINGS.persist_directory
        )

    return db

# ===============================
# LLM
# ===============================
@st.cache_resource
def load_llm():
    if ChatGroq is None:
        raise RuntimeError("langchain-groq not installed")

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama3-8b-8192"),
        temperature=0.85,  # more human
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

# ===============================
# SMART RAG (IMPROVED)
# ===============================
@st.cache_resource
def load_qa_chain():
    llm = load_llm()
    db = load_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k": 6})

    class SmartRAG:
        def __init__(self):
            self.history = []  # conversation history

        def summarize_chunks(self, docs):
            """Summarize PDF chunks for better understanding."""
            summaries = []
            for d in docs:
                prompt = f"""
Summarize this text in your own words, focusing on the key points. 
Do NOT copy sentences verbatim. Be concise.

Text:
{d.page_content}

Summary:
"""
                summary = llm.predict(prompt)
                summaries.append(summary.strip())
            return "\n".join(summaries)

        def generate_prompt(self, query, context):
            """Generate LLM prompt including history to avoid repetition."""
            previous_answers = [a for _, a in self.history[-10:]]
            previous_text = ""
            if previous_answers:
                previous_text = "Previous answers:\n" + "\n".join(f"- {a}" for a in previous_answers) + "\n"

            if len(context.strip()) < 300:
                # Weak context → generic conversational fallback
                prompt = f"""
You are a friendly AI assistant for Wasla Solutions.
- Respond naturally to greetings
- If the question is unclear, ask politely
- Do NOT invent services, history, or claims
- Keep answers short and human

User:
{query}

Answer:
"""
            else:
                prompt = f"""
You are a professional AI assistant for Wasla Solutions.
- Use the context to answer accurately
- Rewrite everything in your own words, do NOT copy verbatim
- Avoid repeating previous answers
- Keep answers human-friendly

{previous_text}
Context:
{context}

User question:
{query}

Answer:
"""
            return prompt

        def __call__(self, query):
            docs = retriever.get_relevant_documents(query)
            context = self.summarize_chunks(docs)
            prompt = self.generate_prompt(query, context)
            answer = llm.predict(prompt)
            self.history.append((query, answer))
            return answer, docs

    return SmartRAG()

# ===============================
# SAVE HISTORY
# ===============================
def save_to_csv(q, a):
    with open("chat_history.csv", "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([q, a])

# ===============================
# STREAMLIT APP
# ===============================
def main():
    st.title("Wasla Solutions – AI PDF Chatbot")

    if "history" not in st.session_state:
        st.session_state.history = []

    question = st.text_area(
        "Ask a question",
        height=140,
        placeholder="Ask anything related to the documents..."
    )

    if st.button("Submit"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Thinking..."):
            qa = load_qa_chain()
            answer, sources = qa(question)

            st.session_state.history.append((question, answer))
            save_to_csv(question, answer)

        st.subheader("✅ Answer")
        st.write(answer)

        with st.expander("📄 Source Context"):
            for i, doc in enumerate(sources, 1):
                st.markdown(f"**Chunk {i}**")
                st.write(doc.page_content[:700] + "...")
                st.markdown("---")

    st.subheader("🧠 Chat History")
    for q, a in reversed(st.session_state.history):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.markdown("---")

if __name__ == "__main__":
    main()
