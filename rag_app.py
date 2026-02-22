"""
Context-Aware RAG Chatbot using LangChain + OpenRouter
Supports: PDF files, text files, URLs, and raw text input

OpenRouter gives you access to 100+ models (GPT-4o, Claude, Llama, Mistral…)
via a single OpenAI-compatible API.  Set your key in .env or as an env var:
    OPENROUTER_API_KEY=sk-or-...
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()  # reads .env file if present

# ── LangChain core ──────────────────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
    DirectoryLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# OpenRouter is OpenAI-compatible → use langchain_openai with a custom base_url
from langchain_openai import ChatOpenAI


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

# ── OpenRouter settings ──────────────────────────────────────────────────────
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Any model slug from https://openrouter.ai/models works here, e.g.:
#   "openai/gpt-4o-mini"          (cheap, fast)
#   "anthropic/claude-3-haiku"    (Anthropic via OpenRouter)
#   "meta-llama/llama-3.1-8b-instruct:free"  (free tier)
#   "mistralai/mistral-7b-instruct:free"      (free tier)
OPENROUTER_MODEL    = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

# ── RAG settings ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"  # free, local
VECTOR_STORE_PATH = "vector_store"
CHUNK_SIZE        = 1000
CHUNK_OVERLAP     = 200
TOP_K_RETRIEVAL   = 4
MEMORY_WINDOW     = 5          # keep last N turns in memory


# ─────────────────────────────────────────────────────────────────────────────
#  Custom RAG Prompt
# ─────────────────────────────────────────────────────────────────────────────

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "input"],
    template="""You are a helpful, knowledgeable assistant.
Use ONLY the following retrieved context to answer the question.
If the answer is not contained in the context, say "I don't have enough information in the provided documents to answer that."

--- CONTEXT ---
{context}
---------------

Question: {input}

Answer (be concise and cite relevant details from the context):"""
)
# ─────────────────────────────────────────────────────────────────────────────
#  Document Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_documents(source: str) -> List[Document]:
    """Load documents from a PDF, text file, directory, URL, or raw string."""
    path = Path(source) if not source.startswith("http") else None

    if source.startswith("http"):
        print(f"  🌐  Loading URL: {source}")
        loader = WebBaseLoader(source)

    elif path and path.is_dir():
        print(f"  📁  Loading directory: {source}")
        loader = DirectoryLoader(
            source,
            glob="**/*.{pdf,txt,md}",
            show_progress=True,
        )

    elif path and path.suffix.lower() == ".pdf":
        print(f"  📄  Loading PDF: {source}")
        loader = PyPDFLoader(str(path))

    elif path and path.suffix.lower() in {".txt", ".md"}:
        print(f"  📝  Loading text file: {source}")
        loader = TextLoader(str(path), encoding="utf-8")

    else:
        # Treat as raw text
        print("  ✏️   Using raw text input")
        return [Document(page_content=source, metadata={"source": "raw_input"})]

    return loader.load()


# ─────────────────────────────────────────────────────────────────────────────
#  Vector Store
# ─────────────────────────────────────────────────────────────────────────────

def build_vector_store(documents: List[Document], embeddings) -> FAISS:
    """Chunk documents and build a FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"  ✂️   Split into {len(chunks)} chunks")

    print("  🔢  Building vector embeddings (this may take a moment)…")
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(VECTOR_STORE_PATH)
    print(f"  💾  Vector store saved to '{VECTOR_STORE_PATH}/'")
    return store


def load_vector_store(embeddings) -> Optional[FAISS]:
    """Load an existing FAISS vector store if available."""
    if Path(VECTOR_STORE_PATH).exists():
        print(f"  📦  Loading existing vector store from '{VECTOR_STORE_PATH}/'")
        return FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  RAG Chain
# ─────────────────────────────────────────────────────────────────────────────

def build_rag_chain(vector_store, llm):
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K_RETRIEVAL, "fetch_k": 10},
    )

    # Combine documents chain (modern replacement)
    question_answer_chain = create_stuff_documents_chain(
        llm, RAG_PROMPT
    )

    # Retrieval chain
    chain = create_retrieval_chain(
        retriever,
        question_answer_chain,
    )

    return chain


# ─────────────────────────────────────────────────────────────────────────────
#  Interactive Chat Loop
# ─────────────────────────────────────────────────────────────────────────────

def chat_loop(chain):
    """Run the interactive Q&A session."""
    print("\n" + "═" * 60)
    print("  🤖  RAG Chatbot ready!  (type 'quit' or 'exit' to stop)")
    print("═" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋  Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("👋  Goodbye!")
            break

        result = chain.invoke({"input": user_input})

        print(f"\n🤖  Assistant:\n{result['answer']}\n")

        # Show source documents
        sources = result.get("source_documents", [])
        if sources:
            seen = set()
            print("  📚  Sources:")
            for doc in sources:
                src = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "")
                label = f"{src}" + (f"  (page {page + 1})" if page != "" else "")
                if label not in seen:
                    print(f"     • {label}")
                    seen.add(label)
        print()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 60)
    print("  📖  Context-Aware RAG Chatbot  |  Powered by LangChain")
    print("═" * 60)

    # ── Pick data source ─────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        source = sys.argv[1]
    else:
        source = input(
            "\nEnter a data source (PDF path / text file / URL / directory)\n"
            "  or press Enter to use the built-in demo text: "
        ).strip()

        if not source:
            source = DEMO_TEXT

    # ── Embeddings ───────────────────────────────────────────────────────────
    print(f"\n⚙️   Initialising embeddings ({EMBEDDING_MODEL})…")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # ── Vector store ─────────────────────────────────────────────────────────
    force_rebuild = source != DEMO_TEXT or not Path(VECTOR_STORE_PATH).exists()

    if force_rebuild:
        print("\n📥  Loading & indexing documents…")
        documents = load_documents(source)
        print(f"  📄  Loaded {len(documents)} document(s)")
        vector_store = build_vector_store(documents, embeddings)
    else:
        vector_store = load_vector_store(embeddings) or build_vector_store(
            load_documents(source), embeddings
        )

    # ── LLM via OpenRouter ────────────────────────────────────────────────────
    print("\n🧠  Connecting to OpenRouter…")
    if not OPENROUTER_API_KEY:
        print("  ❌  OPENROUTER_API_KEY is not set.")
        print("      Add it to a .env file or export it as an environment variable:")
        print("      export OPENROUTER_API_KEY=sk-or-...")
        sys.exit(1)

    llm = ChatOpenAI(
        model=OPENROUTER_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=0.1,
        default_headers={
            # Optional but recommended by OpenRouter for analytics / rate-limit tiers
            "HTTP-Referer": "https://github.com/your-app",  # change to your site
            "X-Title": "RAG Chatbot",
        },
    )
    print(f"  ✅  Connected to OpenRouter  →  model: {OPENROUTER_MODEL}")

    # ── Build chain ───────────────────────────────────────────────────────────
    print("\n🔗  Building RAG chain…")
    chain = build_rag_chain(vector_store, llm)

    # ── Chat ──────────────────────────────────────────────────────────────────
    chat_loop(chain)


# ─────────────────────────────────────────────────────────────────────────────
#  Demo text & Mock LLM (used when Ollama is unavailable)
# ─────────────────────────────────────────────────────────────────────────────

DEMO_TEXT = """
LangChain is a framework for developing applications powered by language models.
It enables applications that are context-aware and reason about their environment.

Key LangChain components:
- **Models**: LLMs, Chat Models, and Embeddings
- **Prompts**: PromptTemplates, FewShotPrompts, Example Selectors
- **Chains**: Sequences of calls to models or utilities
- **Agents**: LLMs that decide which tools to use dynamically
- **Memory**: Persist state between chain runs
- **Retrievers**: Fetch relevant documents from a vector store

RAG (Retrieval-Augmented Generation) combines retrieval with generation:
1. Documents are chunked and embedded into a vector store (e.g., FAISS, Chroma).
2. At query time, semantically similar chunks are retrieved.
3. The retrieved context is passed to an LLM which generates a grounded answer.

FAISS (Facebook AI Similarity Search) is a library for efficient similarity search
over dense vectors. It supports both CPU and GPU and scales to billions of vectors.

Sentence Transformers provide state-of-the-art text embeddings. The model
'all-MiniLM-L6-v2' is 22 MB, runs on CPU, and produces 384-dimensional embeddings.
It is well-suited for semantic search and RAG pipelines.

OpenRouter provides a unified API to access 100+ AI models including GPT-4o,
Claude 3, Llama 3, Mistral, and many more. Get your key at https://openrouter.ai
"""


if __name__ == "__main__":
    main()
