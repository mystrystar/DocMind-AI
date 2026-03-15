# DocMind AI

Full-stack **AI-powered Document Q&A and Semantic Search** application. Upload PDFs, ask questions grounded in the document, and search by meaning.

## Project overview

- **Backend**: Python, FastAPI, ChromaDB, PDF extraction (pdfplumber), token-based chunking (tiktoken). **Embeddings**: Ollama `nomic-embed-text`. **Chat**: Ollama via Semantic Kernel (`OllamaChatCompletion`) with `llama3.2` or `mistral`. No OpenAI or API keys.
- **Frontend**: React (Vite), Tailwind CSS, Axios. Dark theme, responsive layout.
- **Flow**: Upload PDF → extract text → chunk (500 tokens, 50 overlap) → embed with Ollama → store in ChromaDB. Chat and search use semantic retrieval + RAG.

## Architecture (ASCII)

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                     FRONTEND (React)                     │
                    │  Sidebar │ SearchBar │ ChatWindow │ UploadModal          │
                    └───────────────────────────┬─────────────────────────────┘
                                                │ HTTP (localhost:5173 or 3000 → 8000)
                    ┌───────────────────────────▼─────────────────────────────┐
                    │                     BACKEND (FastAPI)                      │
                    │  POST /upload   POST /chat   GET /search   GET /documents  │
                    └───┬─────────────┬────────────┬────────────┬───────────────┘
                        │             │            │            │
         ┌──────────────▼──┐   ┌──────▼──────┐  ┌──▼─────────┐  │
         │ document_service│   │ vector_service│  │ semantic_  │  │ document_store
         │ (pdfplumber,    │   │ (ChromaDB,   │  │ kernel_    │  │ (JSON metadata)
         │  tiktoken chunk)│            │  Ollama     │  │ service    │  │
         └────────────────┘   │ Ollama embed)│  │ (Ollama   │  └──────────────
                               └──────────────┘  │  RAG)      │
                                                 └────────────┘
```

## Setup

### Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **Ollama** installed and running at http://localhost:11434 ([ollama.com](https://ollama.com))

### Ollama models

Pull the required models (one-time):

```bash
ollama pull nomic-embed-text   # for embeddings
ollama pull llama3.2           # for chat (or: ollama pull mistral)
```

### Backend

**Windows (PowerShell):**

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Optional: edit .env to set OLLAMA_HOST, OLLAMA_EMBEDDING_MODEL, OLLAMA_CHAT_MODEL
# Keep the venv activated (you should see (.venv) in the prompt), then run:
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**macOS/Linux:**

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Optional: edit .env for OLLAMA_HOST, OLLAMA_EMBEDDING_MODEL, OLLAMA_CHAT_MODEL
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API: http://localhost:8000  
Docs: http://localhost:8000/docs

### Frontend

```bash
cd frontend
npm install
npm run dev
```

App: http://localhost:5173 (Vite default; or 3000 if configured)

### Sample PDF

```bash
cd sample-data
pip install reportlab
python generate_sample_pdf.py
```

Then upload `sample-data/sample-document.pdf` via the app.

## How it works (RAG pipeline)

1. **Upload**  
   User uploads a PDF. The backend extracts text (pdfplumber), splits it into chunks of **500 tokens** with **50-token overlap** (tiktoken). Each chunk is embedded with **Ollama** (`nomic-embed-text`) and stored in a **ChromaDB** collection (one per document). Document metadata is saved in a JSON file.

2. **Chat (Q&A)**  
   User asks a question and selects a document. The query is embedded with Ollama and **ChromaDB** returns the **top 5** most similar chunks. Those chunks are sent as context to **Ollama** via Semantic Kernel (**OllamaChatCompletion**, model `llama3.2` or `mistral`), with a prompt to answer only from context and say “I don’t know” when insufficient. The reply is returned with **source citations** (chunk index + snippet).

3. **Semantic search**  
   User types in the search bar (with a document selected). The query is embedded with Ollama and **ChromaDB** runs semantic search for that document’s collection. The **top 5** results are returned with a **relevance score** and a **snippet**; the UI shows score badges (green >0.8, yellow >0.6, red below).

4. **Ollama only**  
   No OpenAI or API keys. Set `OLLAMA_HOST` (default `http://localhost:11434`), `OLLAMA_EMBEDDING_MODEL` (default `nomic-embed-text`), and `OLLAMA_CHAT_MODEL` (default `llama3.2`) in `backend/.env` if needed.

## License

MIT.
