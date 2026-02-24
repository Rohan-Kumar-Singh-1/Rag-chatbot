"""
Advanced Multi-Source RAG Chat
Hybrid Search + Re-ranking

Run:
streamlit run streamlit_app.py

Requires:
OPENROUTER_API_KEY
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

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from sentence_transformers import CrossEncoder
from langchain.schema import BaseRetriever

# ─────────────────────────────────
# Page Setup
# ─────────────────────────────────

st.set_page_config(page_title="Advanced RAG Chat", page_icon="🧠")
st.title("🧠 Advanced RAG Chat")
st.caption("Hybrid Search + Re-ranking")

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    st.error("OPENROUTER_API_KEY not set")
    st.stop()


# ─────────────────────────────────
# Session State
# ─────────────────────────────────

if "chain" not in st.session_state:
    st.session_state.chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ─────────────────────────────────
# File Upload
# ─────────────────────────────────

uploaded_files = st.file_uploader(
    "Upload Documents",
    type=["pdf","txt","csv","docx","md"],
    accept_multiple_files=True
)


# ─────────────────────────────────
# URL Input
# ─────────────────────────────────

url_input = st.text_input("Enter Website URL")


# ─────────────────────────────────
# Website Scraper
# ─────────────────────────────────

def scrape_website(url):

    try:

        response = requests.get(url,timeout=10)

        soup = BeautifulSoup(response.text,"html.parser")

        for tag in soup(["script","style","nav","footer"]):
            tag.decompose()

        text = soup.get_text(separator="\n")

        clean_text="\n".join(
            line.strip()
            for line in text.splitlines()
            if line.strip()
        )

        return [Document(page_content=clean_text)]

    except Exception as e:

        st.error(f"Scraping Error: {e}")

        return []


# ─────────────────────────────────
# File Loader
# ─────────────────────────────────

def load_file_documents(files):

    all_docs=[]

    for uploaded_file in files:

        ext=uploaded_file.name.split(".")[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False,suffix=f".{ext}") as tmp:

            tmp.write(uploaded_file.read())

            path=tmp.name


        if ext=="pdf":

            loader=PyPDFLoader(path)
            docs=loader.load()
            all_docs.extend(docs)


        elif ext in ["txt","md"]:

            loader=TextLoader(path,encoding="utf-8")
            docs=loader.load()
            all_docs.extend(docs)


        elif ext=="docx":

            loader=Docx2txtLoader(path)
            docs=loader.load()
            all_docs.extend(docs)


        elif ext=="csv":

            import pandas as pd

            try:
                df=pd.read_csv(path)
            except:
                df=pd.read_csv(path,encoding="latin-1")

            docs=[Document(page_content=df.to_string())]

            all_docs.extend(docs)

    return all_docs


# ─────────────────────────────────
# Re-ranking Function
# ─────────────────────────────────

def rerank_documents(query,docs,reranker,top_k=4):

    pairs=[(query,doc.page_content) for doc in docs]

    scores=reranker.predict(pairs)

    scored=list(zip(scores,docs))

    scored.sort(key=lambda x:x[0],reverse=True)

    return [doc for _,doc in scored[:top_k]]


# ─────────────────────────────────
# Reranker Retriever
# ─────────────────────────────────

from langchain.schema import BaseRetriever
from typing import List
from langchain.schema import Document

class RerankRetriever(BaseRetriever):

    retriever: BaseRetriever
    reranker: object
    top_k: int = 4


    def _get_relevant_documents(self, query: str) -> List[Document]:

        docs = self.retriever.get_relevant_documents(query)

        pairs = [(query, doc.page_content) for doc in docs]

        scores = self.reranker.predict(pairs)

        scored_docs = list(zip(scores, docs))

        scored_docs.sort(key=lambda x: x[0], reverse=True)

        return [doc for _, doc in scored_docs[:self.top_k]]
# ─────────────────────────────────
# Build Chain
# ─────────────────────────────────

def build_chain_from_docs(all_docs):

    st.write("Splitting documents...")

    splitter=RecursiveCharacterTextSplitter(

        chunk_size=1000,
        chunk_overlap=200

    )

    chunks=splitter.split_documents(all_docs)

    st.write(f"Chunks created: {len(chunks)}")


    st.write("Loading embeddings...")

    embeddings=HuggingFaceEmbeddings(

        model_name="sentence-transformers/all-MiniLM-L6-v2",

        model_kwargs={"device":"cpu"},

        encode_kwargs={"normalize_embeddings":True}

    )


    st.write("Building vector store...")

    vectorstore=FAISS.from_documents(chunks,embeddings)


    # FAISS Retriever

    faiss_retriever=vectorstore.as_retriever(

        search_kwargs={"k":8}

    )


    # BM25 Retriever

    bm25_retriever=BM25Retriever.from_documents(chunks)

    bm25_retriever.k=8


    # Hybrid Retriever

    hybrid_retriever=EnsembleRetriever(

        retrievers=[bm25_retriever,faiss_retriever],

        weights=[0.5,0.5]

    )


    st.write("Loading reranker model...")

    reranker=CrossEncoder(

        "cross-encoder/ms-marco-MiniLM-L-6-v2"

    )


    retriever = RerankRetriever(
    retriever=hybrid_retriever,
    reranker=reranker,
    top_k=4
    )

    os.environ["OPENAI_API_KEY"]=api_key
    os.environ["OPENAI_API_BASE"]="https://openrouter.ai/api/v1"


    llm=ChatOpenAI(

        model="openai/gpt-4o-mini",

        temperature=0.1

    )


    memory=ConversationBufferMemory(

        memory_key="chat_history",

        return_messages=True

    )


    return ConversationalRetrievalChain.from_llm(

        llm=llm,

        retriever=retriever,

        memory=memory,

        return_source_documents=False

    )


# ─────────────────────────────────
# Build Chain Trigger
# ─────────────────────────────────

if (uploaded_files or url_input) and st.session_state.chain is None:

    with st.spinner("Processing..."):

        docs1=[]
        docs2=[]

        if uploaded_files:
            docs1=load_file_documents(uploaded_files)

        if url_input:
            docs2=scrape_website(url_input)

        all_docs=docs1+docs2


        if all_docs:

            st.session_state.chain=build_chain_from_docs(all_docs)

            st.success("RAG Ready")

        else:

            st.error("No content loaded")


# ─────────────────────────────────
# Clear Session
# ─────────────────────────────────

if st.button("Clear Session"):

    st.session_state.chain=None

    st.session_state.messages=[]

    st.rerun()


# ─────────────────────────────────
# Chat Interface
# ─────────────────────────────────

if st.session_state.chain:

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):

            st.markdown(msg["content"])


    if prompt:=st.chat_input("Ask a question"):


        st.session_state.messages.append(

            {"role":"user","content":prompt}

        )


        with st.chat_message("user"):

            st.markdown(prompt)


        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):


                result=st.session_state.chain.invoke(

                    {"question":prompt}

                )


                answer=result["answer"]

                st.markdown(answer)


        st.session_state.messages.append(

            {"role":"assistant","content":answer}

        )

else:

    st.info("Upload files or enter a URL")
