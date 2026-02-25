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
import sqlite3
from datetime import datetime

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

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from sentence_transformers import CrossEncoder
from langchain.schema import BaseRetriever
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

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

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

if "user_id" not in st.session_state:
    st.session_state.user_id = None

#
# ─────────────────────────────────
# SQLite Persistence
# ─────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(__file__), "rag_chats.db")


def get_db_connection():

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():

    with get_db_connection() as conn:

        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                source TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
            )
            """
        )

        conn.commit()


def get_or_create_user(username: str) -> int:

    username = username.strip()

    with get_db_connection() as conn:

        cur = conn.cursor()

        cur.execute(
            "SELECT id FROM users WHERE username = ?",
            (username,),
        )
        row = cur.fetchone()

        if row:
            return row["id"]

        cur.execute(
            "INSERT INTO users (username) VALUES (?)",
            (username,),
        )
        conn.commit()
        return cur.lastrowid


def create_chat(user_id: int, title: str = "New Chat") -> int:

    now = datetime.utcnow().isoformat()

    with get_db_connection() as conn:

        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO chats (user_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, title, now, now),
        )
        conn.commit()
        return cur.lastrowid


def list_chats(user_id: int):

    with get_db_connection() as conn:

        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, title, updated_at
            FROM chats
            WHERE user_id = ?
            ORDER BY datetime(updated_at) DESC
            """,
            (user_id,),
        )
        return cur.fetchall()


def save_message(chat_id: int, role: str, content: str):

    now = datetime.utcnow().isoformat()

    with get_db_connection() as conn:

        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO messages (chat_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (chat_id, role, content, now),
        )

        cur.execute(
            "UPDATE chats SET updated_at = ? WHERE id = ?",
            (now, chat_id),
        )

        conn.commit()


def load_messages(chat_id: int):

    with get_db_connection() as conn:

        cur = conn.cursor()
        cur.execute(
            """
            SELECT role, content
            FROM messages
            WHERE chat_id = ?
            ORDER BY id ASC
            """,
            (chat_id,),
        )
        rows = cur.fetchall()

    return [{"role": row["role"], "content": row["content"]} for row in rows]


def save_knowledge_docs(chat_id: int, docs):

    now = datetime.utcnow().isoformat()

    with get_db_connection() as conn:

        cur = conn.cursor()

        for doc in docs:

            source = doc.metadata.get("source", "") if hasattr(doc, "metadata") else ""

            cur.execute(
                """
                INSERT INTO knowledge (chat_id, content, source, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (chat_id, doc.page_content, source, now),
            )

        cur.execute(
            "UPDATE chats SET updated_at = ? WHERE id = ?",
            (now, chat_id),
        )

        conn.commit()


def load_knowledge_docs(chat_id: int):

    with get_db_connection() as conn:

        cur = conn.cursor()
        cur.execute(
            """
            SELECT content
            FROM knowledge
            WHERE chat_id = ?
            ORDER BY id ASC
            """,
            (chat_id,),
        )
        rows = cur.fetchall()

    return [Document(page_content=row["content"]) for row in rows]


def delete_chat_and_data(chat_id: int):

    with get_db_connection() as conn:

        cur = conn.cursor()
        cur.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        cur.execute("DELETE FROM knowledge WHERE chat_id = ?", (chat_id,))
        cur.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        conn.commit()


def maybe_update_chat_title(chat_id: int, first_user_message: str):

    trimmed = first_user_message.strip().splitlines()[0]
    title = " ".join(trimmed.split()[:8])

    with get_db_connection() as conn:

        cur = conn.cursor()
        cur.execute("SELECT title FROM chats WHERE id = ?", (chat_id,))
        row = cur.fetchone()

        if not row:
            return

        existing = (row["title"] or "").strip()

        if existing and existing != "New Chat":
            return

        now = datetime.utcnow().isoformat()
        cur.execute(
            "UPDATE chats SET title = ?, updated_at = ? WHERE id = ?",
            (title, now, chat_id),
        )
        conn.commit()


init_db()


# ─────────────────────────────────
# Sidebar: User and Chat Selection
# ─────────────────────────────────

with st.sidebar:

    st.header("Conversations")

    username = st.text_input("Username", key="username_input")

    if not username.strip():

        st.info("Enter a username to start chatting.")

    else:

        user_id = get_or_create_user(username)
        st.session_state.user_id = user_id

        if st.button("➕ New Chat"):

            new_chat_id = create_chat(user_id)
            st.session_state.current_chat_id = new_chat_id
            st.session_state.messages = []
            st.session_state.chain = None

        chats = list_chats(user_id)

        for chat in chats:

            cols = st.columns([4, 1])

            with cols[0]:

                label = chat["title"] or "Untitled Chat"

                if st.button(label, key=f"chat_{chat['id']}"):

                    st.session_state.current_chat_id = chat["id"]
                    st.session_state.messages = load_messages(chat["id"])
                    st.session_state.chain = None

            with cols[1]:

                if st.button("🗑", key=f"delete_{chat['id']}"):

                    delete_chat_and_data(chat["id"])

                    if st.session_state.current_chat_id == chat["id"]:

                        st.session_state.current_chat_id = None
                        st.session_state.messages = []
                        st.session_state.chain = None

                    st.rerun()

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

current_chat_id = st.session_state.current_chat_id

if not st.session_state.user_id:

    st.info("Set a username in the sidebar to begin.")

elif not current_chat_id:

    st.info("Create or select a chat from the sidebar to attach a knowledge base.")


# ─────────────────────────────────
# Website Scraper
# ─────────────────────────────────

def scrape_website(url):

    try:

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        text = soup.get_text(separator="\n")

        clean_text = "\n".join(
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
    
    
class StreamHandler(BaseCallbackHandler):

    def __init__(self, container):
        self.container = container
        self.text = ""


    def on_llm_new_token(self, token, **kwargs):
        self.text += token
        self.container.markdown(self.text)

CONDENSE_QUESTION_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""
Given the following conversation and a follow up question,
rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}

Follow Up Question:
{question}

Standalone Question:
"""
)
# ─────────────────────────────────
# Build Chain
# ─────────────────────────────────
# ─────────────────────────────────
# Embedding Loader (Cached)
# ─────────────────────────────────

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device":"cpu"},
        encode_kwargs={"normalize_embeddings":True}
    )

@st.cache_resource
def load_reranker():

    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

def build_chain_from_docs(all_docs):

    st.write("Splitting documents...")

    splitter=RecursiveCharacterTextSplitter(

        chunk_size=1000,
        chunk_overlap=200

    )

    st.write("Loading embeddings...")

    embeddings = load_embeddings()

    if all_docs and all_docs[0].metadata.get("source", "").endswith(".csv"):

        vectorstore = FAISS.from_documents(all_docs, embeddings)

    else:

        chunks=splitter.split_documents(all_docs)

        st.write("Building vector store...")

        vectorstore=FAISS.from_documents(chunks,embeddings)


    # FAISS Retriever

    faiss_retriever=vectorstore.as_retriever(

        search_kwargs={"k":20}

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

    reranker = load_reranker()


    retriever = RerankRetriever(
    retriever=hybrid_retriever,
    reranker=reranker,
    top_k=4
    )

    os.environ["OPENAI_API_KEY"]=api_key
    os.environ["OPENAI_API_BASE"]="https://openrouter.ai/api/v1"


    llm=ChatOpenAI(

        model="openai/gpt-4o-mini",

        temperature=0.1,
    
        streaming=True
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        return_source_documents=False
    )

# ─────────────────────────────────
# Build Chain Trigger
# ─────────────────────────────────

if current_chat_id and (uploaded_files or url_input) and st.session_state.chain is None:

    with st.spinner("Processing..."):

        docs1=[]
        docs2=[]

        if uploaded_files:
            docs1=load_file_documents(uploaded_files)

        if url_input:
            docs2=scrape_website(url_input)

        all_docs=docs1+docs2


        if all_docs:

            save_knowledge_docs(current_chat_id, all_docs)

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

# ─────────────────────────────────
# Chat Interface
# ─────────────────────────────────

if current_chat_id and st.session_state.chain:

    # Show chat history
    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):

            st.markdown(msg["content"])


    # New user message
    if prompt := st.chat_input("Ask a question"):

        # Build chat history from previous turns so the chain
        # can answer contextually based on the full conversation.
        chat_history = []
        previous_messages = st.session_state.messages

        for i in range(0, len(previous_messages), 2):
            if (
                i + 1 < len(previous_messages)
                and previous_messages[i]["role"] == "user"
                and previous_messages[i + 1]["role"] == "assistant"
            ):
                chat_history.append(
                    (
                        previous_messages[i]["content"],
                        previous_messages[i + 1]["content"],
                    )
                )


        st.session_state.messages.append(
            {"role":"user","content":prompt}
        )

        save_message(current_chat_id, "user", prompt)


        with st.chat_message("user"):
            st.markdown(prompt)


        # Assistant response with streaming
        with st.chat_message("assistant"):

            message_placeholder = st.empty()

            stream_handler = StreamHandler(message_placeholder)

            result = st.session_state.chain.invoke(
                {"question": prompt, "chat_history": chat_history},
                config={"callbacks":[stream_handler]}
            )

            answer = result["answer"]


        st.session_state.messages.append(
            {"role":"assistant","content":answer}
        )

        save_message(current_chat_id, "assistant", answer)

        maybe_update_chat_title(current_chat_id, prompt)

elif current_chat_id and not st.session_state.chain:

    # Try to load any existing knowledge base for this chat
    existing_docs = load_knowledge_docs(current_chat_id)

    if existing_docs:

        with st.spinner("Loading knowledge base for this chat..."):

            st.session_state.chain = build_chain_from_docs(existing_docs)

            st.rerun()

    else:

        st.info("Upload files or enter a URL to build a knowledge base for this chat.")

else:

    st.info("Set a username and select a chat to begin.")
