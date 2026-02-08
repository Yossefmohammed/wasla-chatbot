import streamlit as st
import os
import csv
import gc
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from .env file (for local development)
load_dotenv()

# For Streamlit Cloud: load from secrets manager
try:
    if hasattr(st, 'secrets') and st.secrets:
        for key, value in st.secrets.items():
            if key not in os.environ:
                os.environ[key] = str(value)
except Exception:
    pass

# Optimize memory: garbage collection
gc.enable()
gc.collect()

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None
    import logging
    logging.exception("langchain_groq import failed")

# Create an adapter that subclasses LangChain's BaseLanguageModel when available
try:
    from langchain.base_language import BaseLanguageModel
except Exception:
    BaseLanguageModel = None

if BaseLanguageModel is not None:
    class ChatGroqAdapter(BaseLanguageModel):
        def __init__(self, client):
            self.client = client

        def predict(self, prompt: str, **kwargs) -> str:
            return self.client.predict(prompt, **kwargs)

        def predict_messages(self, messages, **kwargs):
            return self.client.predict_messages(messages, **kwargs)

        def generate_prompt(self, prompt, **kwargs):
            return self.client.generate_prompt(prompt, **kwargs)

        def invoke(self, prompt, **kwargs):
            return self.client.invoke(prompt, **kwargs)

        async def apredict(self, prompt: str, **kwargs) -> str:
            return await self.client.apredict(prompt, **kwargs)

        async def apredict_messages(self, messages, **kwargs):
            return await self.client.apredict_messages(messages, **kwargs)

        async def agenerate_prompt(self, prompt, **kwargs):
            return await self.client.agenerate_prompt(prompt, **kwargs)
else:
    # Fallback simple adapter (no BaseLanguageModel subclassing)
    class ChatGroqAdapter:
        def __init__(self, client):
            self.client = client

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
    embedding_model = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    try:
        embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
    except Exception as e:
        st.error(f"❌ Failed to load embeddings model: {e}")
        st.stop()

    persist_dir = CHROMA_SETTINGS.persist_directory

    db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    try:
        if db._collection.count() == 0:
            st.info("📚 Building vector database for the first time...")

            folder_path = "docs"  # <-- changed folder name
            if not os.path.exists(folder_path):
                st.error(f"❌ '{folder_path}/' folder not found.")
                st.stop()

            docs = []
            for file in os.listdir(folder_path):
                if file.lower().endswith(".pdf"):
                    loader = PyPDFLoader(os.path.join(folder_path, file))
                    docs.extend(loader.load())

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150
            )
            docs = splitter.split_documents(docs)

            if len(docs) == 0:
                st.error("❌ No text extracted from PDFs.")
                st.stop()

            db = Chroma.from_documents(
                docs,
                embeddings,
                persist_directory=persist_dir
            )

    except Exception as e:
        st.error(f"❌ Failed to build Chroma DB: {e}")
        st.stop()

    return db

# ===============================
# Load LLM (Groq – CACHED)
# ===============================
@st.cache_resource
def load_llm():
    if ChatGroq is None:
        raise RuntimeError("ChatGroq package not available. Ensure langchain-groq is installed.")

    client = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama3-8b-8192"),
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY"),
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # Wrap in adapter if necessary so LangChain's LLMChain accepts it
    try:
        from langchain.base_language import BaseLanguageModel
        is_instance = isinstance(client, BaseLanguageModel)
    except Exception:
        is_instance = False

    if is_instance:
        return client
    else:
        # return adapter that subclasses BaseLanguageModel (defined above)
        try:
            return ChatGroqAdapter(client)
        except Exception:
            return client

# ===============================
# Load QA Chain (CACHED)
# ===============================
@st.cache_resource
def load_qa_chain():
    llm = load_llm()
    db = load_vectorstore()

    class SimpleRetrievalQA:
        def __init__(self, llm, retriever, prompt_template, k=4):
            self.llm = llm
            self.retriever = retriever
            self.prompt = prompt_template
            self.k = k

        def _call_llm(self, text):
            try:
                if hasattr(self.llm, "predict"):
                    return self.llm.predict(text)
                if hasattr(self.llm, "invoke"):
                    return self.llm.invoke(text)
                if callable(self.llm):
                    return self.llm(text)
                raise RuntimeError("LLM has no callable interface (predict/invoke/__call__)")
            except Exception as e:
                msg = str(e).lower()
                if "decommission" in msg or "decommissioned" in msg or "model" in msg and "decommission" in msg:
                    raise RuntimeError(
                        "Groq model appears decommissioned or unsupported. Set a supported model via the GROQ_MODEL env var and see https://console.groq.com/docs/deprecations for recommendations."
                    ) from e
                raise

        def __call__(self, inputs: dict):
            query = inputs.get("query") or inputs.get("question")
            docs = self.retriever.get_relevant_documents(query)
            context = "\n\n".join([d.page_content for d in docs[: self.k]])
            prompt_text = self.prompt.format(context=context, question=query)
            answer = self._call_llm(prompt_text)
            return {"result": answer, "source_documents": docs}

    retriever = db.as_retriever(search_kwargs={"k": 4})
    return SimpleRetrievalQA(llm=llm, retriever=retriever, prompt_template=WASLA_PROMPT)

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
