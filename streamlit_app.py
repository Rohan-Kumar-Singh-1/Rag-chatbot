"""
Streamlit Web UI for the Context-Aware RAG Chatbot (OpenRouter edition)
Run: streamlit run streamlit_app.py

Set OPENROUTER_API_KEY in a .env file or enter it in the sidebar.
"""

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
import os
st.write("API key exists:", bool(os.getenv("OPENROUTER_API_KEY")))
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200
TOP_K           = 4

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""You are a helpful assistant. Answer using ONLY the provided context.
If the answer isn't in the context, say "I don't have enough information in the provided documents."

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

QUESTION: {question}

ANSWER:""",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model…")
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_documents(source_type: str, source) -> list[Document]:
    if source_type == "PDF Upload":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(source.read())
            tmp = f.name
        return PyPDFLoader(tmp).load()

    elif source_type == "Text Upload":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="wb") as f:
            f.write(source.read())
            tmp = f.name
        return TextLoader(tmp, encoding="utf-8").load()

    elif source_type == "Website URL":
        return WebBaseLoader(source).load()

    elif source_type == "Raw Text":
        return [Document(page_content=source, metadata={"source": "user_input"})]


def build_store(docs: list[Document], embeddings) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, embeddings)


def get_llm(model_name: str, api_key: str):
    """Create a ChatOpenAI client pointed at OpenRouter."""
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1,
        default_headers={
            "HTTP-Referer": "https://github.com/your-app",
            "X-Title": "RAG Chatbot",
        },
    )


def build_chain(store: FAISS, llm):
    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": 10},
    )
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=5,
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": RAG_PROMPT},
        return_source_documents=True,
        verbose=False,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    # -- Data source
    st.subheader("📂 Data Source")
    source_type = st.selectbox(
        "Source type",
        ["PDF Upload", "Text Upload", "Website URL", "Raw Text"],
    )

    source_input = None
    if source_type in ("PDF Upload", "Text Upload"):
        accept = "application/pdf" if source_type == "PDF Upload" else "text/plain"
        source_input = st.file_uploader(
            f"Upload {source_type.split()[0]}",
            type=["pdf"] if "PDF" in source_type else ["txt", "md"],
        )
    elif source_type == "Website URL":
        source_input = st.text_input("Enter URL", placeholder="https://example.com")
    else:
        source_input = st.text_area(
            "Paste text",
            height=150,
            placeholder="Paste any text here…",
        )

    # -- OpenRouter settings
    st.subheader("🔑 OpenRouter")

    # Load API key securely from Streamlit Secrets
    api_key = st.secrets["OPENROUTER_API_KEY"]

    st.success("✅ OpenRouter API key loaded securely.")

    # Popular model options on OpenRouter
    MODEL_OPTIONS = [
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
        "anthropic/claude-3-haiku",
        "anthropic/claude-3.5-sonnet",
        "meta-llama/llama-3.1-8b-instruct:free",
        "mistralai/mistral-7b-instruct:free",
        "google/gemma-2-9b-it:free",
        "custom (type below)",
    ]
    model_choice = st.selectbox("Model", MODEL_OPTIONS)
    if model_choice == "custom (type below)":
        model_name = st.text_input("Model slug", placeholder="provider/model-name")
    else:
        model_name = model_choice
        st.caption(f"🔗 [View on OpenRouter](https://openrouter.ai/models/{model_name})")

    # -- Build button
    build_btn = st.button("🚀 Build Knowledge Base", use_container_width=True)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chain = None
        st.rerun()


# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🤖 Context-Aware RAG Chatbot")
st.caption("Powered by LangChain · FAISS · Sentence Transformers · OpenRouter")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "doc_info" not in st.session_state:
    st.session_state.doc_info = ""

# ── Build knowledge base ──────────────────────────────────────────────────────
if build_btn:
    if not source_input:
        st.sidebar.error("⚠️ Please provide a data source first.")
    elif not api_key:
        st.sidebar.error("⚠️ Please enter your OpenRouter API key.")
    elif not model_name:
        st.sidebar.error("⚠️ Please enter a model slug.")
    else:
        with st.spinner("📥 Loading & indexing documents…"):
            try:
                embeddings = get_embeddings()
                docs = load_documents(source_type, source_input)
                store = build_store(docs, embeddings)
                llm = get_llm(model_name, api_key)
                st.session_state.chain = build_chain(store, llm)
                st.session_state.messages = []

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
                )
                n_chunks = len(splitter.split_documents(docs))
                st.session_state.doc_info = (
                    f"✅ Indexed **{len(docs)} document(s)** → **{n_chunks} chunks** "
                    f"| Model: `{model_name}` via OpenRouter | Embeddings: `all-MiniLM-L6-v2`"
                )
                st.sidebar.success("Knowledge base ready!")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {e}")

# Status bar
if st.session_state.doc_info:
    st.info(st.session_state.doc_info)
elif not st.session_state.chain:
    st.warning("👈 Upload a document and click **Build Knowledge Base** to start.")

# ── Chat interface ────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- `{s}`")

if prompt := st.chat_input(
    "Ask anything about your document…",
    disabled=st.session_state.chain is None,
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = st.session_state.chain.invoke({"question": prompt})
            answer = result["answer"]
            sources_raw = result.get("source_documents", [])
            seen, sources = set(), []
            for doc in sources_raw:
                label = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "")
                label += f"  (p.{page + 1})" if page != "" else ""
                if label not in seen:
                    sources.append(label)
                    seen.add(label)

        st.markdown(answer)
        if sources:
            with st.expander("📚 Sources"):
                for s in sources:
                    st.markdown(f"- `{s}`")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
