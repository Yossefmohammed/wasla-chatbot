import streamlit as st
import os
import csv
import gc
import hashlib
import json
import time
import random
import traceback
import requests
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple
from functools import wraps

# ===============================
# ENV & INIT
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

# ===============================
# IMPORTS WITH ERROR HANDLING
# ===============================
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from constant import CHROMA_SETTINGS
except ImportError:
    from dataclasses import dataclass
    @dataclass
    class CHROMA_SETTINGS:
        persist_directory: str = "./chroma_db"

from sentence_transformers import SentenceTransformer, util

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Wasla Solutions - AI Strategy Consultant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# ADVANCED DARK THEME
# ===============================
def set_dark_theme():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0B1020 0%, #151B2B 100%);
        color: #EAEAF2;
    }
    section.main > div {
        max-width: 1000px;
        margin: auto;
        padding: 2rem;
    }
    textarea {
        background-color: rgba(17, 24, 39, 0.8) !important;
        color: #E5E7EB !important;
        border: 1px solid #374151 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        font-size: 16px !important;
        backdrop-filter: blur(10px);
    }
    button {
        background: linear-gradient(45deg, #2563EB, #3B82F6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.3) !important;
    }
    section[data-testid="stSidebar"] button {
        width: 100%;
        margin: 0.2rem 0;
    }
    .chat-card {
        background: rgba(31, 41, 55, 0.5);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(75, 85, 99, 0.3);
        backdrop-filter: blur(10px);
    }
    .source-badge {
        background: #1F2937;
        color: #9CA3AF;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 0.2rem;
    }
    .feedback-btn {
        background: transparent !important;
        border: 1px solid #4B5563 !important;
        color: #9CA3AF !important;
        width: auto !important;
        padding: 0.3rem 1rem !important;
    }
    .feedback-btn:hover {
        background: #2563EB !important;
        border-color: #2563EB !important;
        color: white !important;
    }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

set_dark_theme()

# ===============================
# 🛡️ API KEY VALIDATION
# ===============================
def validate_groq_api_key(api_key):
    """Test Groq API key validity with a minimal request."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 5
    }
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        if r.status_code == 200:
            return True, "✅ API key is valid."
        else:
            return False, f"❌ Groq API error: {r.status_code} - {r.text}"
    except Exception as e:
        return False, f"❌ Could not reach Groq API: {e}"

# ===============================
# SESSION STATE INIT
# ===============================
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "qa_chain" not in st.session_state:
        with st.spinner("🚀 Initializing AI consultant..."):
            st.session_state.qa_chain = load_qa_chain()
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = set()
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "current_question" not in st.session_state:
        st.session_state.current_question = ""
    if "current_answer" not in st.session_state:
        st.session_state.current_answer = ""
    if "current_sources" not in st.session_state:
        st.session_state.current_sources = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = hashlib.md5(
            str(datetime.now().timestamp()).encode()
        ).hexdigest()

# ===============================
# ENHANCED VECTOR STORE
# ===============================
@st.cache_resource(ttl=3600)
def load_vectorstore():
    embeddings = SentenceTransformerEmbeddings(
        model_name=os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    persist_dir = CHROMA_SETTINGS.persist_directory
    os.makedirs(persist_dir, exist_ok=True)
    try:
        db = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        if db._collection.count() == 0:
            db = load_documents_to_vectorstore(embeddings, persist_dir)
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        db = load_documents_to_vectorstore(embeddings, persist_dir)
    return db

def load_documents_to_vectorstore(embeddings, persist_dir):
    docs = []
    folder_path = "docs"
    os.makedirs(folder_path, exist_ok=True)
    loaders = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.csv': CSVLoader,
        '.docx': UnstructuredWordDocumentLoader,
        '.doc': UnstructuredWordDocumentLoader,
        '.pptx': UnstructuredPowerPointLoader,
        '.ppt': UnstructuredPowerPointLoader,
    }
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        file_ext = os.path.splitext(file)[1].lower()
        if file_ext in loaders:
            try:
                loader_class = loaders[file_ext]
                loader = loader_class(file_path)
                docs.extend(loader.load())
                st.sidebar.success(f"✅ Loaded: {file}")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to load {file}: {str(e)}")
    if not docs:
        sample_content = """Wasla Solutions is a digital strategy consulting firm 
        specializing in AI implementation, digital transformation, and strategic 
        advisory for enterprises. We help businesses leverage cutting-edge technology 
        to solve complex challenges and drive growth."""
        sample_path = os.path.join(folder_path, "sample_company_info.txt")
        with open(sample_path, "w") as f:
            f.write(sample_content)
        loader = TextLoader(sample_path)
        docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    docs = splitter.split_documents(docs)
    db = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=persist_dir
    )
    db.persist()
    return db

# ===============================
# LLM WITH STREAMING + VALIDATION
# ===============================
@st.cache_resource
def load_llm():
    if ChatGroq is None:
        raise RuntimeError("langchain-groq not installed. Please install it with: pip install langchain-groq")
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("❌ GROQ_API_KEY not found in secrets or environment.")
        st.stop()
    
    is_valid, message = validate_groq_api_key(api_key)
    if not is_valid:
        st.error(message)
        st.info("Please check your Groq API key in Streamlit secrets or .env file.")
        st.stop()
    
    model_name = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
    
    return ChatGroq(
        model=model_name,
        temperature=0.6,
        groq_api_key=api_key,
        streaming=True,
        max_tokens=1024
    )

# ===============================
# 🛡️ ROBUST RETRY DECORATOR (no retry on 4xx, including 400)
# ===============================
def retry(max_retries=5, delay=3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    err_str = str(e).lower()
                    
                    if any(x in err_str for x in ["400", "401", "403", "404", "invalid api key", "not found", "model not found"]):
                        raise
                    
                    if "429" in err_str or "rate limit" in err_str or "500" in err_str or "503" in err_str:
                        wait = delay * (2 ** i) + random.uniform(0, 0.5)
                        print(f"⚠️ LLM call failed (attempt {i+1}/{max_retries}): {e}. Retrying in {wait:.1f}s")
                        time.sleep(wait)
                        continue
                    
                    wait = delay * (2 ** i) + random.uniform(0, 0.5)
                    print(f"⚠️ LLM call failed (attempt {i+1}/{max_retries}): {e}. Retrying in {wait:.1f}s")
                    time.sleep(wait)
                    continue
            raise Exception(f"Max retries ({max_retries}) exceeded. Last error: {last_exception}")
        return wrapper
    return decorator

# ===============================
# ENHANCED SMART RAG SYSTEM – STRICT DOCUMENT‑ONLY + CREATIVE
# ===============================
def load_qa_chain():
    llm = load_llm()
    db = load_vectorstore()
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    cheap_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=256
    )
    
    class EnhancedSmartRAG:
        def __init__(self, cheap_llm):
            self.history = []
            self.embed_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            self.greetings_db = self._load_greetings()
            self.cheap_llm = cheap_llm
            self.main_llm = llm
        
        @retry(max_retries=5, delay=3)
        def safe_llm_invoke(self, prompt, model="main"):
            if model == "cheap":
                return self.cheap_llm.invoke(prompt).content
            return self.main_llm.invoke(prompt).content
        
        def _load_greetings(self):
            return [
                "hi", "hello", "hey", "greetings", "good morning", 
                "good afternoon", "good evening", "howdy", "hi there",
                "hello there", "what's up", "whats up", "sup",
                "how are you", "how are you doing", "how's it going",
                "how is it going", "how do you do", "nice to meet you"
            ]
        
        def detect_intent(self, query):
            q = query.lower().strip()
            if q in self.greetings_db:
                return "greeting", 1.0
            for greeting in self.greetings_db[:10]:
                if greeting in q:
                    return "greeting", 0.9
            small_talk_patterns = [
                "who are you", "what are you", "what can you do",
                "your name", "capabilities", "help me with",
                "what do you know", "how do you work"
            ]
            for pattern in small_talk_patterns:
                if pattern in q:
                    return "small_talk", 0.9
            return "business", 0.8
        
        # ---------- RAW CONTEXT PREVIEW (NO LLM) – FAST & FACTUAL ----------
        def summarize_chunks(self, docs):
            if not docs:
                return ""
            summaries = []
            for i, d in enumerate(docs[:2]):
                source = d.metadata.get('source', 'Unknown')
                source_name = os.path.basename(source) if source != 'Unknown' else f'Document {i+1}'
                preview = d.page_content[:300].replace('\n', ' ') + "..."
                summaries.append(f"[{source_name}] {preview}")
            return "\n\n".join(summaries)
        
        def check_repetition(self, query) -> Tuple[bool, int]:
            if not self.history:
                return False, 0
            q_emb = self.embed_model.encode(query, convert_to_tensor=True)
            similar_count = 0
            is_repeat = False
            for i, (prev_q, _, prev_emb, _) in enumerate(self.history[-8:]):
                similarity = util.pytorch_cos_sim(q_emb, prev_emb).item()
                recency_boost = 1 - (i / 20)
                threshold = 0.78 + recency_boost
                if similarity > threshold:
                    similar_count += 1
                    if similarity > 0.85:
                        is_repeat = True
            return is_repeat, similar_count
        
        # ---------- STRICT PROMPT – ONLY FROM CONTEXT + CREATIVITY RULES ----------
        def generate_prompt(self, query, context, history_context="", 
                           repeat=False, repeat_count=0):
            
            base_instruction = """You are a consultant representing Wasla Solutions.

STRICT RULES – YOU MUST FOLLOW THEM EXACTLY:
1. ONLY use information from the provided CONTEXT section.
2. If the CONTEXT does NOT contain the answer, say ONLY:
   "I don't have information about that in my knowledge base."
   (If the context has partial information, state what is known first, then acknowledge the gap.)
3. NEVER use your own knowledge, training data, or any external information.
4. NEVER mention Wasla's location, team size, founding date, or any other fact unless it appears in the CONTEXT.
5. Always refer to Wasla Solutions as "we", "us", or "our".
6. Maximum 5 sentences.
7. End with ONE focused follow-up question (only if you actually answered something).

🔥 CREATIVITY & ENGAGEMENT RULES (CRITICAL FOR USER FEEDBACK):
8. Be warm, conversational, and enthusiastic – like a senior consultant who genuinely loves their work.
9. Vary your sentence structure and vocabulary – do NOT repeat the same phrases across different answers.
10. Use rhetorical questions, mild emphasis, and confident language (e.g., "Great question!", "Exactly!", "Here's what I'd suggest…").
11. Never sound like a robot – avoid bullet points, numbered lists, or overly formal phrasing unless it's a direct quote from context.
12. If you are repeating an answer (repeat=True), you MUST use completely different wording and offer a fresh angle or example – never copy the previous response."""

            if repeat_count >= 2:
                return f"""{base_instruction}

The user has asked about this topic multiple times. Do NOT repeat previous answers.

Previous answers:
{history_context}

STRONG INSTRUCTION:
- Acknowledge we've covered this.
- Offer ONE completely new angle, insight, or example (ONLY from CONTEXT).
- Ask a specific, more advanced follow-up question.

Question: {query}

Consultant response:"""
            if repeat:
                return f"""{base_instruction}

The user is asking a similar question again.

Previous answers:
{history_context}

STRONG INSTRUCTION:
- Use completely different wording – do not reuse phrases from previous answers.
- Add a new example, metaphor, or perspective (ONLY from CONTEXT).
- Keep it concise (3-4 sentences).

Question: {query}

New answer:"""
            return f"""{base_instruction}

{history_context}

CONTEXT:
{context}

User question:
{query}

Consultant response:"""
        
        def add_inline_citations(self, answer: str, docs: List) -> str:
            if not docs:
                return answer
            citation_lines = ["\n\n**Sources:**"]
            seen = set()
            for i, doc in enumerate(docs[:3], 1):
                source = doc.metadata.get('source', 'Unknown')
                source_name = os.path.basename(source) if source != 'Unknown' else f'Document {i}'
                if source_name not in seen:
                    citation_lines.append(f"- [{i}] {source_name}")
                    seen.add(source_name)
            return answer + "\n".join(citation_lines)
        
        # ---------- MAIN CALL – STRICT GREETING DETECTION, VARIETY, "HOW ARE YOU" HANDLER ----------
        def __call__(self, query, callback=None):
            intent, confidence = self.detect_intent(query)
            timestamp = datetime.now().isoformat()
            q_lower = query.lower().strip()
            words = q_lower.split()
            
            # ---------- GREETING MODE – ONLY TRIGGER FOR ACTUAL GREETINGS ----------
            # Strict rule: only if query is essentially a greeting (≤3 words and high confidence)
            if intent == "greeting" and confidence > 0.7 and len(words) <= 3:
                # Special handler for "how are you" – brief, friendly, varies
                if any(phrase in q_lower for phrase in ["how are you", "how's it going", "how are things"]):
                    prompts = [
                        "We're doing well, thank you! How can we assist you today?",
                        "All good here, thanks for asking! What can we help you with?",
                        "Doing great – ready to tackle some digital challenges. What's on your mind?"
                    ]
                    answer = random.choice(prompts)
                    if callback: callback(answer)
                    return answer, [], intent
                
                # Generic greeting – vary response each time
                prompt = f"""You are the digital front desk of Wasla Solutions.
Be warm, brief, and professional. Maximum 7 words. 
Vary your greeting – do not repeat the same phrase.
Do not mention being an AI.

User: {query}
Response:"""
                try:
                    answer = self.safe_llm_invoke(prompt)
                    if len(answer.split()) > 8:
                        answer = random.choice([
                            "Wasla Solutions – how can we help?",
                            "Welcome to Wasla! What can we do for you?",
                            "Hi there – Wasla here, ready to help."
                        ])
                except:
                    answer = random.choice([
                        "Wasla Solutions – how can we help?",
                        "Welcome to Wasla! How may we assist?",
                        "Hi! Wasla here – what can we do for you today?"
                    ])
                if callback: callback(answer)
                return answer, [], intent

            # ---------- SMALL TALK MODE – BRANDED FOR IDENTITY, NEUTRAL FOR OTHERS ----------
            if intent == "small_talk" and confidence > 0.7:
                # Branded answer for "who are you / what can you do"
                if any(phrase in q_lower for phrase in ["who are you", "what are you", "what can you do", "your name"]):
                    answer = random.choice([
                        "We're Wasla Solutions – a digital strategy consultancy. We specialise in AI, digital transformation, and growth strategy. How can we help you today?",
                        "Great question! We're Wasla Solutions, your digital strategy partner. We help businesses leverage AI, transform digitally, and scale. What brings you here?",
                        "We're Wasla – a team of digital strategists. Think of us as your co‑pilot for AI, digital products, and business growth. What challenges are you facing?"
                    ])
                    if callback: callback(answer)
                    return answer, [], intent
                
                # Location questions → handled by business mode (RAG)
                if any(phrase in q_lower for phrase in ["where are you", "your location", "headquarters", "based"]):
                    pass  # fall through to business mode
                
                # Generic small talk – keep neutral, no company claims
                else:
                    prompt = f"""You are a helpful assistant.
One short sentence. Do not mention any company details.
Be friendly and natural.

User: {query}
Response:"""
                    try:
                        answer = self.safe_llm_invoke(prompt)
                    except:
                        answer = "I'm here to help with questions about our services."
                    if callback: callback(answer)
                    return answer, [], intent

            # ---------- BUSINESS MODE – STRICT RAG ONLY ----------
            try:
                docs = retriever.get_relevant_documents(query)[:3]
                
                if not docs:
                    answer = "I don't have information about that in my knowledge base."
                    if callback: callback(answer)
                    return answer, [], intent
                
                context = self.summarize_chunks(docs)
                is_repeat, repeat_count = self.check_repetition(query)
                
                history_context = ""
                if self.history:
                    recent = self.history[-2:]
                    history_context = "Previous conversation:\n"
                    for q, a, _, _ in recent:
                        a_short = a[:200] + "…" if len(a) > 200 else a
                        history_context += f"Q: {q}\nA: {a_short}\n\n"
                
                prompt = self.generate_prompt(
                    query, context, history_context, 
                    repeat=is_repeat, repeat_count=repeat_count
                )
                
                answer = None
                try:
                    answer = self.safe_llm_invoke(prompt)
                except Exception as e:
                    print(f"❌ Main LLM failed: {e}. Trying cheap model...")
                    try:
                        answer = self.safe_llm_invoke(prompt, model="cheap")
                    except Exception as e2:
                        print(f"❌ Cheap LLM also failed: {e2}.")
                        answer = "I don't have information about that at the moment. Please try again later."
                
                if answer and "I don't have information" not in answer:
                    answer = self.add_inline_citations(answer, docs)
                
                q_emb = self.embed_model.encode(query, convert_to_tensor=True)
                self.history.append((query, answer, q_emb, timestamp))
                if len(self.history) > 30:
                    self.history = self.history[-30:]
                
                if callback: callback(answer)
                return answer, docs, intent
                
            except Exception as e:
                print("❌ BUSINESS MODE UNRECOVERABLE ERROR:")
                traceback.print_exc()
                error_msg = "I encountered an unexpected issue. Please try again or contact support."
                return error_msg, [], intent
    
    return EnhancedSmartRAG(cheap_llm)

# ===============================
# ENHANCED DATA PERSISTENCE
# ===============================
def save_conversation(question, answer, intent="business", feedback=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs("data", exist_ok=True)
    
    file_exists = os.path.isfile("data/conversations.csv")
    with open("data/conversations.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "question", "answer", "intent", "feedback"])
        writer.writerow([timestamp, question, answer, intent, feedback or ""])
    
    json_file = "data/conversations.json"
    conversation = {
        "timestamp": timestamp,
        "question": question,
        "answer": answer,
        "intent": intent,
        "feedback": feedback,
        "session_id": st.session_state.get("session_id", "")
    }
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            conversations = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        conversations = []
    conversations.append(conversation)
    if len(conversations) > 1000:
        conversations = conversations[-1000:]
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(conversations, f, indent=2)

def save_feedback(question, feedback_type):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs("data", exist_ok=True)
    file_exists = os.path.isfile("data/feedback.csv")
    with open("data/feedback.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "question", "feedback"])
        writer.writerow([timestamp, question, feedback_type])

# ===============================
# SIDEBAR COMPONENTS
# ===============================
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🚀 Wasla Solutions")
        st.markdown("---")
        st.markdown("### 📁 Document Management")
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['pdf', 'txt', 'csv', 'docx', 'doc', 'pptx', 'ppt'],
            accept_multiple_files=True,
            key="doc_uploader"
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                save_path = os.path.join("docs", uploaded_file.name)
                os.makedirs("docs", exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"✅ {uploaded_file.name} uploaded")
            st.cache_resource.clear()
            st.session_state.qa_chain = load_qa_chain()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            user_messages = [m for m in st.session_state.messages if m.get("role") == "user"]
            st.metric("Questions", len(user_messages))
        with col2:
            st.metric("Session ID", st.session_state.get("session_id", "N/A")[:8])
        
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.feedback_given = set()
            st.rerun()
        
        if st.session_state.messages:
            chat_entries = []
            for m in st.session_state.messages:
                if m["role"] == "user":
                    chat_entries.append(f"Q: {m['content']}")
                elif m["role"] == "assistant":
                    if chat_entries and not chat_entries[-1].startswith("A:"):
                        chat_entries[-1] += f"\nA: {m['content']}"
            chat_text = "\n\n".join(chat_entries)
            st.download_button(
                "📥 Export Chat",
                chat_text,
                file_name=f"wasla_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                use_container_width=True
            )

# ===============================
# MAIN CHAT INTERFACE
# ===============================
def main():
    init_session_state()
    render_sidebar()
    
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 2rem;'>"
        "🤖 Wasla AI Strategy Consultant</h1>",
        unsafe_allow_html=True
    )
    
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander("📚 View Sources", expanded=False):
                    for i, source in enumerate(message["sources"][:3], 1):
                        source_name = os.path.basename(
                            source.metadata.get('source', f'document_{i}')
                        )
                        st.markdown(f"**Document {i}:** `{source_name}`")
                        st.markdown(f"```\n{source.page_content[:300]}...\n```")
            if message["role"] == "assistant" and "id" in message:
                msg_id = message["id"]
                if msg_id not in st.session_state.feedback_given:
                    col1, col2, col3 = st.columns([1, 1, 20])
                    with col1:
                        if st.button("👍", key=f"like_{msg_id}_{idx}"):
                            save_feedback(st.session_state.current_question, "positive")
                            st.session_state.feedback_given.add(msg_id)
                            st.rerun()
                    with col2:
                        if st.button("👎", key=f"dislike_{msg_id}_{idx}"):
                            save_feedback(st.session_state.current_question, "negative")
                            st.session_state.feedback_given.add(msg_id)
                            st.rerun()
    
    prompt = st.chat_input("Ask about Wasla's services, digital strategy, or business challenges...")
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            def stream_callback(text):
                message_placeholder.markdown(text + "▌")
            
            try:
                with st.spinner("🤔 Analyzing..."):
                    answer, sources, intent = st.session_state.qa_chain(
                        prompt, callback=stream_callback
                    )
                message_placeholder.markdown(answer)
                if sources:
                    with st.expander("📚 Sources", expanded=False):
                        for i, source in enumerate(sources[:3], 1):
                            source_name = os.path.basename(
                                source.metadata.get('source', f'document_{i}')
                            )
                            st.markdown(f"**{i}. {source_name}**")
                            st.caption(source.page_content[:200] + "...")
                message_id = hashlib.md5(
                    f"{prompt}{datetime.now()}{len(st.session_state.messages)}".encode()
                ).hexdigest()
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "intent": intent,
                    "id": message_id
                })
                st.session_state.current_question = prompt
                st.session_state.current_answer = answer
                st.session_state.current_sources = sources
                save_conversation(prompt, answer, intent)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Please try again or contact support.")
    
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #6B7280; font-size: 0.8rem;'>"
        "Powered by Wasla Solutions • AI Strategy Consultant • "
        "Responses are AI-generated and should be reviewed by a human consultant"
        "</p>",
        unsafe_allow_html=True
    )

# ===============================
# CONSTANT.PY HANDLING
# ===============================
if __name__ == "__main__":
    if not os.path.exists("constant.py"):
        with open("constant.py", "w") as f:
            f.write('''from dataclasses import dataclass

@dataclass
class CHROMA_SETTINGS:
    persist_directory: str = "./chroma_db"
''')
    main()