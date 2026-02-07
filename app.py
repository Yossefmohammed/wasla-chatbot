import streamlit as st
import os
import csv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

from constant import CHROMA_SETTINGS

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Wasla Solutions",
    layout="wide"
)

# ===============================
# Custom Prompt
# ===============================
WASLA_PROMPT = PromptTemplate(
    template="""
You are an assistant answering questions based ONLY on the provided context.

Context:
{context}

Question:
{question}

Answer in a clear, concise, and factual way:
""",
    input_variables=["context", "question"]
)

# ===============================
# Dark Theme
# ===============================
def set_dark_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0B1020;
            color: #EAEAF2;
        }
        section.main > div {
            max-width: 900px;
            margin: auto;
        }
        h1 {
            text-align: center;
            font-size: 42px;
            font-weight: 700;
            color: #FFFFFF;
        }
        textarea {
            background-color: #111827;
            color: #E5E7EB;
            border-radius: 10px;
        }
        button {
            background-color: #2563EB !important;
            color: white !important;
            border-radius: 10px;
            width: 100%;
        }
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True
    )

set_dark_theme()

# ===============================
# Load Vector DB (CACHED)
# ===============================
@st.cache_resource
def load_vectorstore():
    embeddings = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    db = Chroma(
        persist_directory=CHROMA_SETTINGS.persist_directory,
        embedding_function=embeddings
    )
    return db

# ===============================
# Load LLM (Groq – CACHED)
# ===============================
@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama3-8b-8192",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )

# ===============================
# Load QA Chain (CACHED)
# ===============================
@st.cache_resource
def load_qa_chain():
    llm = load_llm()
    db = load_vectorstore()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 4}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": WASLA_PROMPT},
        return_source_documents=True
    )
    return qa

# ===============================
# Save Chat History
# ===============================
def save_to_csv(question, answer):
    with open("chat_history.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([question, answer])

# ===============================
# Streamlit App
# ===============================
def main():
    st.title("Wasla Solutions – Chatbot Feedback")

    if "history" not in st.session_state:
        st.session_state.history = []

    question = st.text_area(
        "Ask a question about the feedback PDFs",
        height=140,
        placeholder="Type your question here..."
    )

    if st.button("Submit"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Thinking..."):
            qa = load_qa_chain()
            result = qa({"query": question})

            answer = result["result"]
            source_docs = result["source_documents"]

            st.session_state.history.append({
                "q": question,
                "a": answer
            })

            save_to_csv(question, answer)

        st.subheader("✅ Answer")
        st.write(answer)

        with st.expander("📄 Source Documents"):
            for i, doc in enumerate(source_docs, 1):
                st.markdown(f"**Document {i}:**")
                st.write(doc.page_content[:800] + "...")
                st.markdown("---")

    # ===============================
    # Chat History
    # ===============================
    st.subheader("🧠 Chat History")
    for item in reversed(st.session_state.history):
        st.markdown(f"**Q:** {item['q']}")
        st.markdown(f"**A:** {item['a']}")
        st.markdown("---")

if __name__ == "__main__":
    main()
