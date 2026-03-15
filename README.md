# DocMind AI

Full-stack **AI-powered Document Q&A and Semantic Search** application. Upload PDFs, ask questions grounded in the document, and search by meaning.

## Project overview

- **Backend**: Python, FastAPI, ChromaDB, PDF extraction (pdfplumber), token-based chunking (tiktoken). **Embeddings**: local (ChromaDB default, free) or OpenAI. **Chat**: Ollama (free, local) or OpenAI GPT-4o via Semantic Kernel.
- **Frontend**: React (Vite), Tailwind CSS, Axios. Dark theme, responsive layout.
- **Flow**: Upload PDF → extract text → chunk (500 tokens, 50 overlap) → embed → store in ChromaDB. Chat and search use semantic retrieval + RAG.

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
         │  tiktoken chunk)│   │  OpenAI      │  │ service    │  │
         └────────────────┘   │  embeddings) │  │ (GPT-4o    │  └──────────────
                               └──────────────┘  │  RAG)      │
                                                 └────────────┘
```

## Setup

### Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **OpenAI API key** (optional) — only needed if you don’t use the free tier

### Free tier (no OpenAI quota)

You can run DocMind AI **without an OpenAI API key**:

1. **Embeddings**: set `USE_LOCAL_EMBEDDINGS=true` in `backend/.env`. The app will use ChromaDB’s default local model (all-MiniLM-L6-v2) on your machine — no API calls.
2. **Chat**: set `OLLAMA_BASE_URL=http://localhost:11434` in `backend/.env` and install [Ollama](https://ollama.com). Run `ollama pull llama3.1` (or another model). Chat will use your local model instead of GPT-4o.

Copy `backend/.env.example` to `backend/.env` and set the free-tier options above; leave `OPENAI_API_KEY` commented out.

### Backend

**Windows (PowerShell):**

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Edit .env: for free tier set USE_LOCAL_EMBEDDINGS=true and OLLAMA_BASE_URL=http://localhost:11434
# Or set OPENAI_API_KEY=sk-... for OpenAI
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
# Edit .env and set OPENAI_API_KEY=sk-...
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
   User uploads a PDF. The backend extracts text (pdfplumber), splits it into chunks of **500 tokens** with **50-token overlap** (tiktoken). Each chunk is embedded and stored in a **ChromaDB** collection (one per document). With **free tier** (`USE_LOCAL_EMBEDDINGS=true`), ChromaDB’s default local model (all-MiniLM-L6-v2) is used; otherwise OpenAI embeddings are used. Document metadata is saved in a JSON file.

2. **Chat (Q&A)**  
   User asks a question and selects a document. The query is embedded and **ChromaDB** returns the **top 5** most similar chunks. Those chunks are sent as context to either **Ollama** (free, local) or **OpenAI GPT-4o** via Semantic Kernel, with a prompt to answer only from context and say “I don’t know” when insufficient. The reply is returned with **source citations** (chunk index + snippet).

3. **Semantic search**  
   User types in the search bar (with a document selected). The query is embedded and **ChromaDB** runs semantic search for that document’s collection. The **top 5** results are returned with a **relevance score** and a **snippet**; the UI shows score badges (green >0.8, yellow >0.6, red below).

4. **Free vs paid**  
   **Free**: `USE_LOCAL_EMBEDDINGS=true` (local embeddings) and `OLLAMA_BASE_URL=http://localhost:11434` (local chat with Ollama). **Paid**: set `OPENAI_API_KEY` and leave the free-tier options unset to use OpenAI for embeddings and chat.

## License

MIT.
