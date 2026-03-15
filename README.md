# DocMind AI

Full-stack **AI-powered Document Q&A and Semantic Search** application. Upload PDFs, ask questions grounded in the document, and search by meaning.

## Project overview

- **Backend**: Python, FastAPI, Semantic Kernel (OpenAI GPT-4o), ChromaDB, PDF extraction (pdfplumber), token-based chunking (tiktoken), OpenAI embeddings.
- **Frontend**: React (Vite), Tailwind CSS, Axios. Dark theme, responsive layout.
- **Flow**: Upload PDF → extract text → chunk (500 tokens, 50 overlap) → embed → store in ChromaDB. Chat and search use semantic retrieval + RAG.

## Architecture (ASCII)

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                     FRONTEND (React)                     │
                    │  Sidebar │ SearchBar │ ChatWindow │ UploadModal          │
                    └───────────────────────────┬─────────────────────────────┘
                                                │ HTTP (localhost:3000 → 8000)
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
- **OpenAI API key** (for GPT-4o and embeddings)

### Backend

**Windows (PowerShell):**

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
# Important: keep the venv activated (you should see (.venv) in the prompt), then run:
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

App: http://localhost:3000

### Sample PDF

```bash
cd sample-data
pip install reportlab
python generate_sample_pdf.py
```

Then upload `sample-data/sample-document.pdf` via the app.

## How it works (RAG pipeline)

1. **Upload**  
   User uploads a PDF. The backend extracts text (pdfplumber), splits it into chunks of **500 tokens** with **50-token overlap** (tiktoken). Each chunk is embedded with **OpenAI text-embedding-3-small** and stored in a **ChromaDB** collection (one per document). Document metadata (filename, upload time, chunk count) is saved in a JSON file.

2. **Chat (Q&A)**  
   User asks a question and selects a document. The query is embedded and **ChromaDB** returns the **top 5** most similar chunks. Those chunks are concatenated as context and sent to **Semantic Kernel** with a system prompt that tells **GPT-4o** to answer **only from the context** and to say **"I don't know"** when the context is insufficient. The reply is returned with **source citations** (chunk index + snippet).

3. **Semantic search**  
   User types in the search bar (with a document selected). The query is embedded and **ChromaDB** runs semantic search for that document’s collection. The **top 5** results are returned with a **relevance score** and a **snippet**; the UI shows score badges (green >0.8, yellow >0.6, red below).

4. **Semantic Kernel**  
   A **Kernel** is created with **OpenAIChatCompletion** (GPT-4o). The RAG prompt is a template with `{{$context}}` and `{{$question}}`, filled via **KernelArguments**. `invoke_prompt` runs the model and the result is parsed to return the assistant’s text.

## License

MIT.
