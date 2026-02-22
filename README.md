# 🧠 Context-Aware RAG Chatbot (LangChain + OpenRouter)

A fully functional **Retrieval-Augmented Generation (RAG) Chatbot** built using **LangChain**, **FAISS**, and **OpenRouter**.

This repository includes:

* 🖥️ CLI-based RAG chatbot
* 🌐 Streamlit Web UI (temporary session-based memory)
* 📄 Support for PDF, text files, directories, URLs, and raw text
* 💾 Persistent vector store (CLI version)
* ⚡ Local embeddings (no API cost)
* 🤖 Access to 100+ LLMs via OpenRouter

---

## 📁 Project Structure

```
.
├── rag_app.py          # Full-featured CLI RAG chatbot
├── streamlit_app.py    # Streamlit web app (temporary memory)
├── requirements.txt    # Project dependencies
└── README.md
```

---

# 🚀 Features

## 1️⃣ CLI RAG Chatbot (`rag_app.py`)

📄 File: 

### ✅ Supports:

* PDF files
* Text files (.txt, .md)
* Directories
* URLs
* Raw text input

### ✅ Persistent Vector Store

* Uses FAISS
* Saves embeddings locally (`vector_store/`)
* Reuses index if available

### ✅ Advanced Retrieval

* MMR retrieval
* Custom RAG prompt
* Source document display
* OpenRouter model switching

### ▶️ Run CLI Version

```bash
python rag_app.py
```

Or pass a source directly:

```bash
python rag_app.py myfile.pdf
```

---

## 2️⃣ Streamlit Web App (`streamlit_app.py`)

📄 File: 

### ✅ Features:

* Upload a PDF
* Chat with it instantly
* Session-based memory
* Temporary vector database
* Clear session button

⚠️ No persistent storage (everything resets after session)

### ▶️ Run Web App

```bash
streamlit run streamlit_app.py
```

---

# 🔑 OpenRouter Setup

This project uses **OpenRouter (OpenAI-compatible API)**.

### Step 1 — Get API Key

Create account at:
👉 [https://openrouter.ai](https://openrouter.ai)

### Step 2 — Set Environment Variable

Linux / Mac:

```bash
export OPENROUTER_API_KEY=sk-or-xxxx
```

Windows:

```bash
set OPENROUTER_API_KEY=sk-or-xxxx
```

Or create a `.env` file:

```
OPENROUTER_API_KEY=sk-or-xxxx
```

---

# 🧩 Supported Models

You can switch models by setting:

```
OPENROUTER_MODEL=openai/gpt-4o-mini
```

Other examples:

* `anthropic/claude-3-haiku`
* `meta-llama/llama-3.1-8b-instruct`
* `mistralai/mistral-7b-instruct`

---

# 🧠 How RAG Works (In This Project)

1. Documents are loaded (PDF, URL, etc.)
2. Text is chunked (1000 tokens with 200 overlap)
3. Chunks are embedded using:

   ```
   sentence-transformers/all-MiniLM-L6-v2
   ```
4. FAISS stores embeddings
5. At query time:

   * Top-K relevant chunks retrieved
   * Passed to LLM with custom RAG prompt
   * Answer generated grounded in context

---

# 📦 Installation

📄 Dependencies: 

## 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
```

## 2️⃣ Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

## 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

# ⚙️ Tech Stack

* LangChain
* FAISS (Vector DB)
* Sentence Transformers (Local embeddings)
* OpenRouter (LLM provider)
* Streamlit (Web UI)
* Python 3.9+

---

# 🧪 Demo Mode (CLI)

If you don’t provide a source, the CLI runs with built-in demo text explaining:

* LangChain
* RAG
* FAISS
* Sentence Transformers
* OpenRouter

---

# 📚 Example Use Cases

* Research document Q&A
* Chat with PDFs
* Company knowledge base
* Personal document assistant
* Study material summarization
* Website content QA

---

# 🔒 Security Notes

* API key is never stored in code
* Uses environment variables
* Embeddings run locally (no embedding API cost)

---

# 📈 Future Improvements

* Add ChromaDB support
* Add authentication for Streamlit
* Dockerize deployment
* Deploy on Render / Railway
* Add multi-user support
* Add hybrid search (BM25 + vector)

---

# 🤝 Contributing

Pull requests are welcome.

If you find this project useful, ⭐ star the repository.

---

# 📜 License

MIT License
