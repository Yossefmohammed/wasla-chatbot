import streamlit as st
import os
import csv
import gc
import hashlib
import json
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from pathlib import Path

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

# Document loaders for multiple formats
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Handle constant import gracefully
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
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0B1020 0%, #151B2B 100%);
        color: #EAEAF2;
    }
    
    /* Container styling */
    section.main > div {
        max-width: 1000px;
        margin: auto;
        padding: 2rem;
    }
    
    /* Text areas */
    textarea {
        background-color: rgba(17, 24, 39, 0.8) !important;
        color: #E5E7EB !important;
        border: 1px solid #374151 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        font-size: 16px !important;
        backdrop-filter: blur(10px);
    }
    
    /* Buttons */
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
    
    /* Sidebar button specific */
    section[data-testid="stSidebar"] button {
        width: 100%;
        margin: 0.2rem 0;
    }
    
    /* Cards for chat history */
    .chat-card {
        background: rgba(31, 41, 55, 0.5);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(75, 85, 99, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Source citations */
    .source-badge {
        background: #1F2937;
        color: #9CA3AF;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 0.2rem;
    }
    
    /* Feedback buttons */
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
# SESSION STATE INIT
# ===============================
def init_session_state():
    """Initialize all session state variables"""
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
# ENHANCED VECTOR STORE WITH MULTI-FORMAT SUPPORT
# ===============================
@st.cache_resource(ttl=3600)
def load_vectorstore():
    """Load or create vector store with support for multiple document formats"""
    embeddings = SentenceTransformerEmbeddings(
        model_name=os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    
    persist_dir = CHROMA_SETTINGS.persist_directory
    
    # Ensure persist directory exists
    os.makedirs(persist_dir, exist_ok=True)
    
    try:
        db = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        
        # If no documents exist, load from docs folder
        if db._collection.count() == 0:
            db = load_documents_to_vectorstore(embeddings, persist_dir)
            
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        # Create new vector store
        db = load_documents_to_vectorstore(embeddings, persist_dir)
    
    return db

def load_documents_to_vectorstore(embeddings, persist_dir):
    """Load documents from various formats into vector store"""
    docs = []
    folder_path = "docs"
    
    # Create docs folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    
    # Supported file extensions and their loaders
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
        # Add a sample document if no docs exist
        sample_content = """Wasla Solutions is a digital strategy consulting firm 
        specializing in AI implementation, digital transformation, and strategic 
        advisory for enterprises. We help businesses leverage cutting-edge technology 
        to solve complex challenges and drive growth."""
        
        sample_path = os.path.join(folder_path, "sample_company_info.txt")
        with open(sample_path, "w") as f:
            f.write(sample_content)
        
        loader = TextLoader(sample_path)
        docs = loader.load()
    
    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    docs = splitter.split_documents(docs)
    
    # Create vector store
    db = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=persist_dir
    )
    db.persist()
    
    return db

# ===============================
# LLM WITH STREAMING
# ===============================
@st.cache_resource
def load_llm():
    """Load LLM with streaming support"""
    if ChatGroq is None:
        raise RuntimeError("langchain-groq not installed. Please install it with: pip install langchain-groq")
    
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama3-70b-8192"),
        temperature=0.6,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        streaming=True,
        max_tokens=1024
    )

# ===============================
# ENHANCED SMART RAG SYSTEM
# ===============================
def load_qa_chain():
    """Load the enhanced QA chain with all improvements"""
    llm = load_llm()
    db = load_vectorstore()
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    class EnhancedSmartRAG:
        def __init__(self):
            self.history = []
            self.embed_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            self.greetings_db = self._load_greetings()
        
        # ---------------------------
        # Enhanced Intent Detection
        # ---------------------------
        def _load_greetings(self):
            """Load expanded greetings database"""
            return [
                "hi", "hello", "hey", "greetings", "good morning", 
                "good afternoon", "good evening", "howdy", "hi there",
                "hello there", "what's up", "whats up", "sup",
                "how are you", "how are you doing", "how's it going",
                "how is it going", "how do you do", "nice to meet you"
            ]
        
        def detect_intent(self, query):
            """Enhanced intent detection with confidence scoring"""
            q = query.lower().strip()
            
            # Check for exact matches
            if q in self.greetings_db:
                return "greeting", 1.0
            
            # Check for partial matches in greetings
            for greeting in self.greetings_db[:10]:
                if greeting in q:
                    return "greeting", 0.9
            
            # Small talk patterns
            small_talk_patterns = [
                "who are you", "what are you", "what can you do",
                "your name", "capabilities", "help me with",
                "what do you know", "how do you work"
            ]
            
            for pattern in small_talk_patterns:
                if pattern in q:
                    return "small_talk", 0.9
            
            return "business", 0.8
        
        # ---------------------------
        # Enhanced Summarization
        # ---------------------------
        def summarize_chunks(self, docs):
            """Improved summarization with better prompt engineering"""
            if not docs:
                return "No relevant documents found."
            
            summaries = []
            
            for i, d in enumerate(docs):
                # Get source filename
                source = d.metadata.get('source', 'Unknown')
                source_name = os.path.basename(source) if source != 'Unknown' else f'Document {i+1}'
                
                prompt = f"""You are an expert consultant. Extract ONLY the key business insights from this text.

Text:
{d.page_content}

Requirements:
- 1-2 sentences maximum
- Focus on actionable information
- Include specific numbers or facts if present
- No fluff or marketing language

Key insights:"""
                
                try:
                    summary = llm.invoke(prompt).content
                    summaries.append(f"[{source_name}] {summary.strip()}")
                except Exception as e:
                    summaries.append(f"[{source_name}] {d.page_content[:100]}...")
            
            return "\n".join(summaries)
        
        # ---------------------------
        # Enhanced Repetition Detection
        # ---------------------------
        def is_repeat_question(self, query):
            """Improved repetition detection with temporal decay"""
            if not self.history:
                return False
            
            q_emb = self.embed_model.encode(query, convert_to_tensor=True)
            
            # Check last 10 questions with decreasing threshold
            for i, (_, _, prev_emb, _) in enumerate(self.history[-10:]):
                similarity = util.pytorch_cos_sim(q_emb, prev_emb).item()
                
                # More recent questions get stricter threshold
                recency_boost = 1 - (i / 20)
                threshold = 0.85 + recency_boost
                
                if similarity > threshold:
                    return True
            return False
        
        # ---------------------------
        # Generate Prompt with History
        # ---------------------------
        def generate_prompt(self, query, context, history_context="", repeat=False):
            """Generate prompt with conversation history"""
            
            if repeat:
                return f"""The user is asking a similar question again.

Previous answers:
{history_context}

Provide a fresh perspective:
- Use completely different wording
- Add a new example or angle
- Keep it concise (3-4 sentences)

Question: {query}

New answer:"""
            
            return f"""You are a senior digital strategy consultant at Wasla Solutions.

STRICT RULES:
- Maximum 5 sentences
- Lead with specific, actionable advice
- Zero marketing fluff or corporate jargon
- End with ONE focused follow-up question
- Use context only if directly relevant

{history_context}

Context:
{context}

User question:
{query}

Consultant response:"""
        
        # ---------------------------
        # Main Call with Streaming
        # ---------------------------
        def __call__(self, query, callback=None):
            """Main entry point with streaming support"""
            
            intent, confidence = self.detect_intent(query)
            timestamp = datetime.now().isoformat()
            
            # GREETING MODE
            if intent == "greeting" and confidence > 0.7:
                prompt = f"""You are a friendly consultant. Respond in ONE sentence.

User: {query}

Response:"""
                try:
                    answer = llm.invoke(prompt).content
                except:
                    answer = "Hello! How can I help you today?"
                
                if callback:
                    callback(answer)
                
                return answer, [], intent
            
            # SMALL TALK MODE
            if intent == "small_talk" and confidence > 0.7:
                prompt = f"""You are a Wasla Solutions AI assistant. Be helpful and brief.

User: {query}

Response (1-2 sentences):"""
                try:
                    answer = llm.invoke(prompt).content
                except:
                    answer = "I'm here to help with business strategy, digital transformation, and AI implementation questions."
                
                if callback:
                    callback(answer)
                
                return answer, [], intent
            
            # BUSINESS MODE
            try:
                # Get relevant documents
                docs = retriever.get_relevant_documents(query)
                
                # Summarize context
                context = self.summarize_chunks(docs) if docs else ""
                
                # Check for repetition
                repeat = self.is_repeat_question(query)
                
                # Build conversation history
                history_context = ""
                if self.history:
                    recent = self.history[-3:]
                    history_context = "Previous conversation:\n"
                    for q, a, _, _ in recent:
                        history_context += f"Q: {q}\nA: {a}\n\n"
                
                # Generate prompt
                prompt = self.generate_prompt(query, context, history_context, repeat)
                
                # Get answer
                answer = llm.invoke(prompt).content
                
                # Store in history
                q_emb = self.embed_model.encode(query, convert_to_tensor=True)
                self.history.append((query, answer, q_emb, timestamp))
                
                # Keep history manageable
                if len(self.history) > 50:
                    self.history = self.history[-50:]
                
                if callback:
                    callback(answer)
                
                return answer, docs, intent
                
            except Exception as e:
                error_msg = f"I encountered an issue while processing your request. Please try again or rephrase your question."
                return error_msg, [], intent
    
    return EnhancedSmartRAG()

# ===============================
# ENHANCED DATA PERSISTENCE
# ===============================
def save_conversation(question, answer, intent="business", feedback=None):
    """Save conversation with metadata"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save to CSV
    file_exists = os.path.isfile("data/conversations.csv")
    with open("data/conversations.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "question", "answer", "intent", "feedback"])
        writer.writerow([timestamp, question, answer, intent, feedback or ""])
    
    # Also save to JSON for easier analysis
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
    
    # Keep last 1000 conversations
    if len(conversations) > 1000:
        conversations = conversations[-1000:]
    
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(conversations, f, indent=2)

def save_feedback(question, feedback_type):
    """Save user feedback"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    os.makedirs("data", exist_ok=True)
    
    file_exists = os.isfile("data/feedback.csv")
    with open("data/feedback.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "question", "feedback"])
        writer.writerow([timestamp, question, feedback_type])

# ===============================
# SIDEBAR COMPONENTS
# ===============================
def render_sidebar():
    """Render sidebar with document management and analytics"""
    
    with st.sidebar:
        st.markdown("## 🚀 Wasla Solutions")
        st.markdown("---")
        
        # Document Management
        st.markdown("### 📁 Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['pdf', 'txt', 'csv', 'docx', 'doc', 'pptx', 'ppt'],
            accept_multiple_files=True,
            key="doc_uploader"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Save uploaded file
                save_path = os.path.join("docs", uploaded_file.name)
                os.makedirs("docs", exist_ok=True)
                
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.success(f"✅ {uploaded_file.name} uploaded")
            
            # Clear cache to reload documents
            st.cache_resource.clear()
            st.session_state.qa_chain = load_qa_chain()
            st.rerun()
        
        st.markdown("---")
        
        # Quick Stats
        st.markdown("### 📊 Session Stats")
        col1, col2 = st.columns(2)
        
        with col1:
            user_messages = [m for m in st.session_state.messages if m.get("role") == "user"]
            st.metric(
                "Questions",
                len(user_messages)
            )
        
        with col2:
            st.metric(
                "Session ID",
                st.session_state.get("session_id", "N/A")[:8],
                help="Unique session identifier"
            )
        
        st.markdown("---")
        
        # Clear Chat
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.feedback_given = set()
            st.rerun()
        
        # Export Chat - FIXED THE BUG HERE
        if st.session_state.messages:
            chat_entries = []
            for m in st.session_state.messages:
                if m["role"] == "user":
                    # Find the corresponding assistant message
                    chat_entries.append(f"Q: {m['content']}")
                elif m["role"] == "assistant":
                    # Add the last user's Q with this A
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
    """Main application with enhanced chat interface"""
    
    # Initialize session
    init_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Main chat area
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 2rem;'>"
        "🤖 Wasla AI Strategy Consultant</h1>",
        unsafe_allow_html=True
    )
    
    # Display chat messages
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander("📚 View Sources", expanded=False):
                    for i, source in enumerate(message["sources"][:3], 1):
                        source_name = os.path.basename(
                            source.metadata.get('source', f'document_{i}')
                        )
                        st.markdown(f"**Document {i}:** `{source_name}`")
                        st.markdown(f"```\n{source.page_content[:300]}...\n```")
            
            # Show feedback buttons for assistant messages
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
    
    # Chat input
    prompt = st.chat_input("Ask about Wasla's services, digital strategy, or business challenges...")
    
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            def stream_callback(text):
                message_placeholder.markdown(text + "▌")
            
            try:
                with st.spinner("🤔 Analyzing..."):
                    answer, sources, intent = st.session_state.qa_chain(
                        prompt, 
                        callback=stream_callback
                    )
                
                # Display final answer
                message_placeholder.markdown(answer)
                
                # Show sources if available
                if sources:
                    with st.expander("📚 Sources", expanded=False):
                        for i, source in enumerate(sources[:3], 1):
                            source_name = os.path.basename(
                                source.metadata.get('source', f'document_{i}')
                            )
                            st.markdown(f"**{i}. {source_name}**")
                            preview = source.page_content[:200] + "..."
                            st.caption(preview)
                
                # Add assistant message to history
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
                
                # Store current for feedback
                st.session_state.current_question = prompt
                st.session_state.current_answer = answer
                st.session_state.current_sources = sources
                
                # Save conversation
                save_conversation(prompt, answer, intent)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Please try again or contact support.")
    
    # Footer
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
    # Create constant.py if it doesn't exist
    if not os.path.exists("constant.py"):
        with open("constant.py", "w") as f:
            f.write('''from dataclasses import dataclass

@dataclass
class CHROMA_SETTINGS:
    persist_directory: str = "./chroma_db"
''')
    
    main()