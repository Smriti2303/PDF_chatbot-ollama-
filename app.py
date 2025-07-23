import os
import time
import pickle
import streamlit as st
from htmltemplates import css, bot_template, user_template
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseRetriever
from langchain.schema.document import Document
from typing import List, Any
from pydantic import Field
from langchain_community.llms import Ollama
import pandas as pd

# --- Load FAISS index ---
if not os.path.exists("faiss_index/index.pkl"):
    from preprocess_pdfs import main as build_index
    build_index()

with open("faiss_index/index.pkl", "rb") as f:
    vectorstore_data = pickle.load(f)

if isinstance(vectorstore_data, tuple) and hasattr(vectorstore_data[0], "similarity_search"):
    vectorstore, tender_data = vectorstore_data
else:
    st.error("❌ FAISS index is invalid or missing.")
    st.stop()

# --- Session State ---
for key, default in {
    "chat_history": [],
    "chat_history_display": [],
    "selected_pdf": "All PDFs",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- Available PDFs ---
try:
    all_docs = vectorstore.similarity_search("dummy", k=1000)
except:
    all_docs = []

all_pdfs = sorted({
    doc.metadata.get("source", "").split("/")[-1]
    for doc in all_docs if "source" in doc.metadata
})

# --- Ollama LLM Setup ---
llm = Ollama(
    model="tinyllama",  # or "mistral", "llama2", etc.
    temperature=0.1
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

class CustomRetriever(BaseRetriever):
    vectorstore: Any = Field()
    selected_pdf: str = Field()

    def _get_relevant_documents(self, query: str) -> List[Document]:
        all_results = self.vectorstore.similarity_search(query, k=20)
        if self.selected_pdf == "All PDFs":
            return all_results[:5]
        return [doc for doc in all_results if doc.metadata.get("source", "").endswith(self.selected_pdf)][:5]

retriever = CustomRetriever(vectorstore=vectorstore, selected_pdf=st.session_state.selected_pdf)
chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

def display_chat(user_msg, bot_msg, response_time=None):
    st.markdown(user_template.replace("{{MSG}}", user_msg), unsafe_allow_html=True)
    msg = bot_msg
    if response_time:
        msg += f"\n\n⏱ Responded in {response_time}s"
    st.markdown(bot_template.replace("{{MSG}}", msg), unsafe_allow_html=True)

def handle_userinput(query: str):
    with st.spinner("🤖 Thinking..."):
        start = time.time()
        result = chain({"question": query, "chat_history": st.session_state.chat_history})
        end = time.time()
        response_time = round(end - start, 2)

        st.session_state.chat_history.append((query, result["answer"]))
        st.session_state.chat_history_display.insert(0, (query, result["answer"], response_time))
        display_chat(query, result["answer"], response_time)

def extract_field(pdf_name: str, query: str) -> str:
    local_retriever = CustomRetriever(vectorstore=vectorstore, selected_pdf=pdf_name)
    local_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=local_retriever,
        memory=ConversationBufferMemory(return_messages=True)
    )
    try:
        result = local_chain.invoke({"question": query, "chat_history": []})
        return result['answer'].strip()
    except:
        return "❓ Not Found"

def compare_tenders(pdf1: str, pdf2: str):
    criteria = {
        "Tender Title": "What is the tender title?",
        "Estimated Cost": "What is the estimated cost of the tender?",
        "EMD": "What is the EMD or Earnest Money Deposit amount?",
        "Completion Time": "What is the completion time or duration of the tender?",
        "Eligibility": "What is the eligibility criteria for this tender?"
    }

    data = {"Criteria": list(criteria.keys())}
    for pdf in [pdf1, pdf2]:
        answers = []
        for field, q in criteria.items():
            answer = extract_field(pdf, q)
            answers.append(answer)
        data[pdf] = answers

    df = pd.DataFrame(data)
    st.markdown("### 🔍 Tender Comparison Table")
    st.table(df)

def main():
    st.set_page_config(page_title="🚆 Tender Chatbot | Offline Ollama", layout="wide")
    st.markdown(css, unsafe_allow_html=True)

    st.markdown("## 📘 Chat with Tender PDFs")
    col1, col2 = st.columns([1.2, 4])

    with col1:
        with st.sidebar:
            st.markdown("### 📂 Select Tender PDF")
            st.session_state.selected_pdf = st.selectbox("Choose PDF", ["All PDFs"] + all_pdfs)

            st.markdown("### 💡 Quick Prompts")
            if st.button("📋 List all tenders"):
                handle_userinput("List all tenders")
            if st.button("🧾 Summarize this tender"):
                handle_userinput("Summarize this tender document")
            if st.button("📌 Eligibility Criteria"):
                handle_userinput("What is the eligibility criteria?")

            st.markdown("### 🧾 Tender Comparison")
            pdf1 = st.selectbox("🔸 Tender A", all_pdfs, key="pdf1")
            pdf2 = st.selectbox("🔹 Tender B", all_pdfs, key="pdf2")
            if st.button("📊 Compare Tenders"):
                compare_tenders(pdf1, pdf2)

            st.markdown("### 💬 Chat History")
            for i, (q, a, t) in enumerate(st.session_state.chat_history_display):
                with st.expander(f"🔹 {q}", expanded=False):
                    st.markdown(f"{a}\n\n_🕓 {t}s_")

    with col2:
        question = st.text_input("💬 Ask your question here")
        if question:
            handle_userinput(question)

        if st.session_state.chat_history_display:
            st.markdown("### 🧠 Chat Conversation")
            for q, a, t in st.session_state.chat_history_display:
                display_chat(q, a, t)

if __name__ == "__main__":
    main()
