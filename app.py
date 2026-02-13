import streamlit as st
import os, csv, gc, hashlib, json, time, random, traceback, requests
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple
from functools import wraps

# ===============================
# ENV & INIT
# ===============================
load_dotenv()
gc.enable()
gc.collect()

# ===============================
# CONSTANTS / SETTINGS
# ===============================
if not os.path.exists("constant.py"):
    with open("constant.py", "w") as f:
        f.write('''from dataclasses import dataclass\n@dataclass\nclass CHROMA_SETTINGS:\n    persist_directory: str = "./chroma_db"\n''')
from constant import CHROMA_SETTINGS

# ===============================
# THEME
# ===============================
def set_dark_theme():
    st.markdown("""<style>
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
    .chat-card{background:rgba(31,41,55,0.5); border-radius:12px; padding:1.5rem; margin:1rem 0; border:1px solid rgba(75,85,99,0.3); backdrop-filter:blur(10px);}
    .source-badge{background:#1F2937; color:#9CA3AF; padding:0.2rem 0.8rem; border-radius:20px; font-size:0.8rem; display:inline-block; margin:0.2rem;}
    .feedback-btn{background:transparent !important; border:1px solid #4B5563 !important; color:#9CA3AF !important; width:auto !important; padding:0.3rem 1rem !important;}
    .feedback-btn:hover{background:#2563EB !important; border-color:#2563EB !important; color:white !important;}
    footer{visibility:hidden;}
    </style>""", unsafe_allow_html=True)

set_dark_theme()

# ===============================
# SESSION STATE INIT
# ===============================
def init_session_state():
    defaults = {
        "messages": [], "qa_chain": None, "feedback_given": set(),
        "uploaded_files": [], "current_question": "", "current_answer": "",
        "current_sources": [], "session_id": hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.qa_chain is None:
        with st.spinner("🚀 Initializing AI consultant..."):
            st.session_state.qa_chain = load_qa_chain()

# ===============================
# RUN APP
# ===============================
def main():
    init_session_state()
    render_sidebar()
    st.markdown("<h1 style='text-align:center; margin-bottom:2rem;'>🤖 Wasla AI Strategy Consultant</h1>", unsafe_allow_html=True)

    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"]=="assistant" and message.get("sources"):
                with st.expander("📚 View Sources", expanded=False):
                    for i, s in enumerate(message["sources"][:3]):
                        name = os.path.basename(s.metadata.get("source", f"doc_{i+1}"))
                        st.markdown(f"**{i+1}. {name}**")
                        st.caption(s.page_content[:200]+"...")

    # Chat input
    user_input = st.chat_input("Ask about Wasla's services, strategy, or business...")
    if user_input:
        st.session_state.messages.append({"role":"user", "content":user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            def callback(text): placeholder.markdown(text+"▌")
            try:
                answer, sources, intent = st.session_state.qa_chain(user_input, callback=callback)
                placeholder.markdown(answer)
                st.session_state.messages.append({
                    "role":"assistant","content":answer,"sources":sources,"intent":intent,
                    "id":hashlib.md5(f"{user_input}{time.time()}".encode()).hexdigest()
                })
                st.session_state.current_question = user_input
                st.session_state.current_answer = answer
                st.session_state.current_sources = sources
                save_conversation(user_input, answer, intent)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    st.markdown("<p style='text-align:center;color:#6B7280;font-size:0.8rem;'>Powered by Wasla Solutions • AI Strategy Consultant • Responses are AI-generated</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
