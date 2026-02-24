"""
Multi-Source RAG Chat (Files + Website Scraping)
Run: streamlit run streamlit_app.py
Requires OPENROUTER_API_KEY
"""

import os
import tempfile
import streamlit as st
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


# ── Page Setup ──────────────────────────────
st.set_page_config(page_title="RAG Chat", page_icon="🧠", layout="centered")
st.title("🧠 Chat With Files or Websites")
st.caption("Upload documents or enter a URL · Session-only vector DB")

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


# ── File Upload ─────────────────────────────
uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "txt", "csv", "docx", "md"],
    accept_multiple_files=True
)

# ── URL Input ───────────────────────────────
url_input = st.text_input("Or enter a website URL to scrape:")


# ── Website Scraper ─────────────────────────
def scrape_website(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove scripts and styles
        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        clean_text = "\n".join(
            [line.strip() for line in text.splitlines() if line.strip()]
        )

        return [Document(page_content=clean_text)]

    except Exception as e:
        st.error(f"Error scraping website: {e}")
        return []


# ── Load Documents From Files ───────────────
def load_file_documents(files):
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

        # CSV (Safe Pandas Version)
        elif file_extension == "csv":
            import pandas as pd

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

    return all_docs


# ── Build RAG Chain ─────────────────────────
def build_chain_from_docs(all_docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

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

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=False,
    )


# ── Build Chain When Content Provided ───────
if (uploaded_files or url_input) and st.session_state.chain is None:
    with st.spinner("Processing content..."):

        docs_from_files = []
        docs_from_url = []

        if uploaded_files:
            docs_from_files = load_file_documents(uploaded_files)

        if url_input:
            docs_from_url = scrape_website(url_input)

        all_docs = docs_from_files + docs_from_url

        if all_docs:
            st.session_state.chain = build_chain_from_docs(all_docs)
            st.success("Content loaded into temporary memory!")
        else:
            st.error("No content could be processed.")


# ── Clear Session ───────────────────────────
if st.button("🗑️ Clear Session"):
    st.session_state.chain = None
    st.session_state.messages = []
    st.rerun()


# ── Chat Interface ──────────────────────────
if st.session_state.chain:

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about your content..."):

        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                result = st.session_state.chain.invoke(
                    {"question": prompt}
                )

                answer = result["answer"]

                st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

else:
    st.info("Upload files or enter a URL to start chatting.")
