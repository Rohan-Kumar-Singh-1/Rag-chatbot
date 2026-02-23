"""
Multi-Document RAG Chat (Temporary Memory)
No persistent DB
Run: streamlit run streamlit_app.py
Requires OPENROUTER_API_KEY
"""

import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# ── Page Setup ──────────────────────────────
st.set_page_config(page_title="Document Chat", page_icon="📂", layout="centered")
st.title("📂 Chat With Your Documents (Temporary Memory)")
st.caption("Supports PDF, CSV, TXT, DOCX, MD · Session-only vector DB")

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

# ── File Upload (Multiple Supported) ───────
uploaded_files = st.file_uploader(
    "Upload your documents",
    type=["pdf", "txt", "csv", "docx", "md"],
    accept_multiple_files=True
)


# ── Build Chain Function ────────────────────
def build_chain(files):

    all_docs = []

    for uploaded_file in files:
        file_extension = uploaded_file.name.split(".")[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # PDF
        if file_extension == "pdf":
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            all_docs.extend(docs)

        # TXT / MD
        elif file_extension in ["txt", "md"]:
            loader = TextLoader(tmp_path, encoding="utf-8")
            docs = loader.load()
            all_docs.extend(docs)

        # CSV (Pandas Safe Version)
        elif file_extension == "csv":
            import pandas as pd
            from langchain.schema import Document

            try:
                df = pd.read_csv(tmp_path, encoding="utf-8")
            except:
                df = pd.read_csv(tmp_path, encoding="latin-1")

            text_content = df.to_string(index=False)
            docs = [Document(page_content=text_content)]
            all_docs.extend(docs)

        # DOCX
        elif file_extension == "docx":
            loader = Docx2txtLoader(tmp_path)
            docs = loader.load()
            all_docs.extend(docs)

        else:
            continue

    if not all_docs:
        return None

    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(all_docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    # LLM setup (OpenRouter)
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


# ── Build Chain on Upload ───────────────────
if uploaded_files and st.session_state.chain is None:
    with st.spinner("Processing documents..."):
        st.session_state.chain = build_chain(uploaded_files)
        st.success("Documents loaded into temporary memory!")


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

    if prompt := st.chat_input("Ask a question about your documents..."):
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
    st.info("Upload one or more documents to start chatting.")
