"""
Temporary PDF RAG Chat
No persistent DB
Run: streamlit run streamlit_app.py
Requires OPENROUTER_API_KEY
"""

import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
# ── Page Setup ──────────────────────────────
st.set_page_config(page_title="PDF Chat", page_icon="📄", layout="centered")
st.title("📄 Chat With Your PDF (Temporary Memory)")
st.caption("No persistent storage · Session-only vector DB")
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.error("OPENROUTER_API_KEY not set.")
    st.stop()

# ── Session State ───────────────────────────
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Upload PDF ──────────────────────────────
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

def build_chain(pdf_file):
    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    # Load PDF
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    # Embeddings (in memory only)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    # LLM
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0.1,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=False,
    )

    return chain

# ── Build Chain ─────────────────────────────
if uploaded_file and st.session_state.chain is None:
    with st.spinner("Processing PDF..."):
        st.session_state.chain = build_chain(uploaded_file)
        st.success("PDF loaded into temporary memory!")

# ── Clear Button ────────────────────────────
if st.button("🗑️ Clear Session"):
    st.session_state.chain = None
    st.session_state.messages = []
    st.rerun()

# ── Chat Interface ──────────────────────────
if st.session_state.chain:

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about the PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.chain.invoke({"question": prompt})
                answer = result["answer"]
            st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

else:
    st.info("Upload a PDF to start chatting.")