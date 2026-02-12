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

try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None

from constant import CHROMA_SETTINGS
from sentence_transformers import SentenceTransformer, util

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
        temperature=0.7,  # balanced
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

# ===============================
# SMART RAG (CONSULTANT MODE)
# ===============================
def load_qa_chain():   # Removed cache for session isolation
    llm = load_llm()
    db = load_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k": 6})

    class SmartRAG:
        def __init__(self):
            self.history = []
            self.embed_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )

        def summarize_chunks(self, docs):
            summaries = []
            for d in docs:
                prompt = f"""
Summarize this text briefly in your own words.
Focus only on key information.
Do NOT copy sentences.

Text:
{d.page_content}

Summary:
"""
                summary = llm.predict(prompt)
                summaries.append(summary.strip())

            return "\n".join(summaries)

        def is_repeat_question(self, query):
            if not self.history:
                return False

            q_emb = self.embed_model.encode(query, convert_to_tensor=True)

            for _, _, prev_emb in self.history[-5:]:
                similarity = util.pytorch_cos_sim(q_emb, prev_emb).item()
                if similarity > 0.85:
                    return True
            return False

        def generate_prompt(self, query, context, repeat=False):

            previous_answers = [a for _, a, _ in self.history[-5:]]
            previous_text = ""
            if previous_answers:
                previous_text = "Previous answers (avoid repeating):\n" + "\n".join(
                    f"- {a}" for a in previous_answers
                ) + "\n"

            if repeat:
                return f"""
You already answered something very similar.
Rephrase the answer differently and shorter.
Avoid repeating structure.

User:
{query}

Answer:
"""

            # Greeting / small talk mode
            if len(context.strip()) < 200:
                return f"""
You are a smart, natural AI assistant representing Wasla Solutions.

Rules:
- If greeting → respond briefly (1 sentence).
- No marketing tone.
- No pushing services unless asked.
- Keep it human and natural.

User:
{query}

Answer:
"""

            # Consultant mode
            return f"""
You are a senior digital strategy consultant at Wasla Solutions.

Rules:
- Max 4–6 sentences.
- Give actionable advice FIRST.
- Avoid marketing language.
- Avoid generic phrases.
- Ask at most ONE focused follow-up question.
- Do not repeat previous answers.
- Be confident and practical.

{previous_text}

Context:
{context}

User question:
{query}

Answer:
"""

        def __call__(self, query):
            docs = retriever.get_relevant_documents(query)
            context = self.summarize_chunks(docs)
            repeat = self.is_repeat_question(query)

            prompt = self.generate_prompt(query, context, repeat)
            answer = llm.predict(prompt)

            q_emb = self.embed_model.encode(query, convert_to_tensor=True)
            self.history.append((query, answer, q_emb))

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
    st.title("Wasla Solutions – AI Strategy Chatbot")

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = load_qa_chain()

    if "history" not in st.session_state:
        st.session_state.history = []

    question = st.text_area(
        "Ask a question",
        height=140,
        placeholder="Ask anything related to Wasla or your business..."
    )

    if st.button("Submit"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Thinking..."):
            answer, sources = st.session_state.qa_chain(question)

            st.session_state.history.append((question, answer))
            save_to_csv(question, answer)

        st.subheader("✅ Answer")
        st.write(answer)

        with st.expander("📄 Source Context"):
            for i, doc in enumerate(sources, 1):
                st.markdown(f"**Chunk {i}**")
                st.write(doc.page_content[:600] + "...")
                st.markdown("---")

    st.subheader("🧠 Chat History")
    for q, a in reversed(st.session_state.history):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.markdown("---")

if __name__ == "__main__":
    main()
