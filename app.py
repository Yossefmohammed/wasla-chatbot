import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import BitsAndBytesConfig, pipeline, AutoTokenizer, AutoModelForCausalLM
from chromadb.config import Settings
import torch
import os
import csv

# Import your constants from constant.py
from constant import CHROMA_SETTINGS

# ===============================
# Custom Prompt Template
# ===============================
WASLA_PROMPT = PromptTemplate(
    template="""Context: {context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

# ===============================
# Dark Theme + Center Layout
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

# ===============================
# Load LLaMA 2 Model
# ===============================
@st.cache_resource()
def load_llm():
    hf_token = os.getenv("HF_TOKEN")
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=hf_token,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.3,
        top_p=0.8,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )

    return HuggingFacePipeline(pipeline=gen_pipeline)

# ===============================
# Safe String Extraction Function
# ===============================
def extract_answer_from_result(result):
    """Safely extract answer from pipeline result"""
    if isinstance(result, list) and len(result) > 0:
        if isinstance(result[0], dict) and 'generated_text' in result[0]:
            return result[0]['generated_text']
        elif isinstance(result[0], str):
            return result[0]
        else:
            return str(result[0])
    elif isinstance(result, str):
        return result
    else:
        return str(result)

# ===============================
# Save to CSV
# ===============================
def save_to_csv(question, answer):
    with open("chat_history.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([question, answer])

# ===============================
# Custom QA Function
# ===============================
def custom_qa(question, llm):
    embeddings = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"}
    )
    
    db = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_SETTINGS.persist_directory
    )
    
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(question)
    
    context = "\n".join([doc.page_content for doc in docs])
    prompt = WASLA_PROMPT.format(context=context, question=question)
    
    result = llm(prompt)
    result_str = extract_answer_from_result(result)

    if "Answer:" in result_str:
        split_parts = result_str.split("Answer:")
        if len(split_parts) > 1:
            answer = split_parts[1].strip()
        else:
            answer = result_str.strip()
    else:
        lines = result_str.split('\n')
        answer_lines = []
        question_found = False
        
        for line in lines:
            if "Question:" in line:
                question_found = True
                continue
            if question_found and line.strip():
                answer_lines.append(line.strip())
        
        answer = ' '.join(answer_lines).strip()
    
    artifacts = [
        "Based on the following information",
        "provide a direct answer",
        "be concise and factual",
        "Context:",
        "Question:",
    ]
    
    for artifact in artifacts:
        if artifact in answer:
            answer = answer.replace(artifact, "").strip()
    
    return answer, docs

# ===============================
# Streamlit App
# ===============================
def main():
    st.set_page_config(page_title="Wasla Solutions", layout="wide")
    set_dark_theme()

    # ====== Session State for Chat History ======
    if "history" not in st.session_state:
        st.session_state.history = []

    st.title("Wasla Solutions – Chatbot Feedback")

    question = st.text_area(
        "Ask a question about the feedback PDFs",
        height=140,
        placeholder="Type your question here..."
    )

    if st.button("Submit"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Processing your question..."):
            llm = load_llm()
            answer, source_docs = custom_qa(question, llm)

            # Save to session history
            st.session_state.history.append({"q": question, "a": answer})

            # Save to CSV
            save_to_csv(question, answer)

        st.subheader("✅ Answer")
        st.write(answer)

        with st.expander("📄 Source Documents"):
            st.write(source_docs)

    # ====== Display Chat History ======
    st.subheader("🧠 Chat History")
    for item in st.session_state.history:
        st.write(f"**Q:** {item['q']}")
        st.write(f"**A:** {item['a']}")
        st.markdown("---")

if __name__ == "__main__":
    main()
