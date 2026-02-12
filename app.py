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
        model=os.getenv("GROQ_MODEL", "llama3-70b-8192"),  # upgraded
        temperature=0.6,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

# ===============================
# SMART RAG SYSTEM
# ===============================
def load_qa_chain():
    llm = load_llm()
    db = load_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k": 5})

    class SmartRAG:
        def __init__(self):
            self.history = []
            self.embed_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )

        # ---------------------------
        # Intent Detection
        # ---------------------------
        def detect_intent(self, query):
            q = query.lower().strip()

            greetings = [
                "hi", "hello", "hey",
                "good morning", "good evening",
                "how are you", "what's up", "whats up"
            ]

            small_talk = [
                "how are you doing",
                "what are you doing",
                "who are you"
            ]

            if q in greetings:
                return "greeting"

            if any(phrase in q for phrase in small_talk):
                return "small_talk"

            return "business"

        # ---------------------------
        # Summarize Retrieved Docs
        # ---------------------------
        def summarize_chunks(self, docs):
            summaries = []
            for d in docs:
                prompt = f"""
Summarize this text briefly.
Focus only on key information.
Do NOT copy sentences.

Text:
{d.page_content}

Summary:
"""
                summary = llm.predict(prompt)
                summaries.append(summary.strip())

            return "\n".join(summaries)

        # ---------------------------
        # Repetition Detection
        # ---------------------------
        def is_repeat_question(self, query):
            if not self.history:
                return False

            q_emb = self.embed_model.encode(query, convert_to_tensor=True)

            for _, _, prev_emb in self.history[-5:]:
                similarity = util.pytorch_cos_sim(q_emb, prev_emb).item()
                if similarity > 0.88:
                    return True
            return False

        # ---------------------------
        # Generate Prompt
        # ---------------------------
        def generate_prompt(self, query, context, repeat=False):

            previous_answers = [a for _, a, _ in self.history[-5:]]

            previous_text = ""
            if previous_answers:
                previous_text = "Avoid repeating these previous answers:\n"
                previous_text += "\n".join(f"- {a}" for a in previous_answers)

            if repeat:
                return f"""
The user asked a similar question before.
Rephrase the answer differently.
Be shorter and avoid same structure.

User:
{query}

Answer:
"""

            return f"""
You are a senior digital strategy consultant at Wasla Solutions.

Rules:
- 4–6 sentences max.
- Give actionable advice first.
- Avoid marketing tone.
- Avoid generic phrases.
- Ask at most ONE focused follow-up question.
- Be practical and confident.
- Use provided context only if relevant.

{previous_text}

Context:
{context}

User question:
{query}

Answer:
"""

        # ---------------------------
        # Main Call
        # ---------------------------
        def __call__(self, query):

            intent = self.detect_intent(query)

            # GREETING MODE
            if intent == "greeting":
                prompt = f"""
Respond naturally and briefly (1 sentence).
Be friendly and human.
Do NOT mention services.

User:
{query}

Answer:
"""
                answer = llm.predict(prompt)
                return answer, []

            # SMALL TALK MODE
            if intent == "small_talk":
                prompt = f"""
Respond conversationally and briefly.
No marketing language.

User:
{query}

Answer:
"""
                answer = llm.predict(prompt)
                return answer, []

            # BUSINESS MODE
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

        if sources:
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
