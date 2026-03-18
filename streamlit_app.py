"""
Streamlit UI for DocMind AI.

This file hosts:
1) Streamlit frontend (docs list, upload, search, chat UI)
2) Optionally the existing FastAPI backend (backend/main.py) in-process

Note:
- Hosting a React/Vite bundle directly on Streamlit Cloud is not straightforward.
  This approach keeps the full app functionality by calling your existing backend API.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import streamlit as st


REPO_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = REPO_ROOT / "backend"

# Allow `from routers import ...` inside backend/main.py to work.
sys.path.insert(0, str(BACKEND_DIR))


DEFAULT_API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
START_BACKEND = os.environ.get("START_BACKEND", "true").strip().lower() in {"1", "true", "yes", "y"}


def _start_backend_if_needed() -> None:
    """
    Start backend/main.py (uvicorn) in a background thread.
    Intended for local runs. On managed hosts you may want START_BACKEND=false.
    """

    if not START_BACKEND:
        return

    # If backend is already up, don't start again.
    import requests  # local import to avoid hard dependency when START_BACKEND=false

    try:
        r = requests.get(f"{DEFAULT_API_URL}/health", timeout=1.0)
        if r.status_code == 200:
            return
    except Exception:
        pass

    try:
        import uvicorn
    except Exception as e:  # pragma: no cover
        st.error(
            "uvicorn not available, so the backend cannot be started automatically.\n\n"
            "Fix: install dependencies from the repo root, then restart Streamlit:\n"
            "  pip install -r requirements.txt\n\n"
            "Or set START_BACKEND=false and run the backend separately on port 8000."
        )
        return

    def _run() -> None:
        # main:app refers to backend/main.py module name `main` (since we inserted backend/ into sys.path).
        config = uvicorn.Config(
            "main:app",
            host="127.0.0.1",
            port=8000,
            log_level="warning",
            # No reload in this embedded mode.
            reload=False,
        )
        server = uvicorn.Server(config)
        server.run()

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    # Wait until /health returns OK (or time out).
    for _ in range(60):
        try:
            r = requests.get(f"{DEFAULT_API_URL}/health", timeout=1.0)
            if r.status_code == 200:
                return
        except Exception:
            time.sleep(0.5)
    st.warning("Backend did not become ready in time. Check logs or set START_BACKEND=false.")


def _api_get(path: str, params: dict[str, Any] | None = None) -> Any:
    import requests

    url = f"{DEFAULT_API_URL}{path}"
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def _api_delete(path: str) -> Any:
    import requests

    url = f"{DEFAULT_API_URL}{path}"
    r = requests.delete(url, timeout=60)
    r.raise_for_status()
    return r.json()


def _api_post_json(path: str, payload: dict[str, Any]) -> Any:
    import requests

    url = f"{DEFAULT_API_URL}{path}"
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()


def _api_upload_pdf(path: str, filename: str, content: bytes) -> Any:
    import requests

    url = f"{DEFAULT_API_URL}{path}"
    files = {"file": (filename, content, "application/pdf")}
    r = requests.post(url, files=files, timeout=600)
    r.raise_for_status()
    return r.json()


def _ensure_state() -> None:
    st.session_state.setdefault("documents", [])
    st.session_state.setdefault("active_doc_id", None)
    st.session_state.setdefault("active_doc_filename", None)
    st.session_state.setdefault("chunks", None)
    st.session_state.setdefault("chat_doc_id", None)
    st.session_state.setdefault("messages", [])


def _refresh_documents() -> None:
    data = _api_get("/documents")
    docs = data.get("documents", [])
    st.session_state["documents"] = docs

    # If active doc was deleted or is missing, clear selection.
    active_id = st.session_state.get("active_doc_id")
    if active_id and not any(d.get("doc_id") == active_id for d in docs):
        st.session_state["active_doc_id"] = None
        st.session_state["active_doc_filename"] = None


def _set_active_doc(doc: dict[str, Any] | None) -> None:
    if not doc:
        st.session_state["active_doc_id"] = None
        st.session_state["active_doc_filename"] = None
        st.session_state["chunks"] = None
        return

    doc_id = doc.get("doc_id")
    st.session_state["active_doc_id"] = doc_id
    st.session_state["active_doc_filename"] = doc.get("filename") or ""
    st.session_state["chunks"] = None

    # Reset chat when document changes.
    if st.session_state.get("chat_doc_id") != doc_id:
        st.session_state["chat_doc_id"] = doc_id
        st.session_state["messages"] = []


def _load_chunks(doc_id: str) -> None:
    data = _api_get(f"/documents/{doc_id}/chunks")
    st.session_state["chunks"] = data.get("chunks", [])


def _delete_doc(doc_id: str) -> None:
    _api_delete(f"/documents/{doc_id}")
    # Refresh list + clear selection if needed.
    _refresh_documents()


def _render_chat() -> None:
    messages: list[dict[str, Any]] = st.session_state.get("messages", [])
    active_doc_id = st.session_state.get("active_doc_id")

    if not active_doc_id:
        st.info("Select a document from the sidebar to start chatting.")
        return

    for msg in messages:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        sources = msg.get("sources", []) or []

        with st.chat_message(role):
            st.write(content)
            if role != "user" and sources:
                with st.expander(f"Sources ({len(sources)})", expanded=False):
                    for s in sources:
                        chunk_index = s.get("chunk_index")
                        score = s.get("score")
                        snippet = s.get("snippet", "")
                        header = f"Chunk {chunk_index + 1}" if isinstance(chunk_index, int) else "Chunk"
                        if score is not None:
                            header += f" • {(score * 100):.0f}%"
                        st.markdown(f"**{header}**")
                        st.write(snippet)

    prompt = st.chat_input("Ask a question...")
    if not prompt:
        return

    # Push user message immediately for responsiveness.
    messages.append({"role": "user", "content": prompt, "sources": []})
    st.session_state["messages"] = messages

    active_doc_id = st.session_state["active_doc_id"]
    with st.spinner("Generating answer..."):
        try:
            data = _api_post_json(
                "/chat",
                payload={
                    "document_id": active_doc_id,
                    "question": prompt,
                    # backend ignores history today, but sending keeps schema-compatible
                    "history": [],
                },
            )
        except Exception as e:
            st.error(f"Chat failed: {e}")
            messages.append(
                {
                    "role": "assistant",
                    "content": "Sorry, something went wrong. Please try again.",
                    "sources": [],
                }
            )
            st.session_state["messages"] = messages
            return

    messages.append(
        {
            "role": "assistant",
            "content": data.get("answer", ""),
            "sources": data.get("sources", []) or [],
        }
    )
    st.session_state["messages"] = messages


def _render_sidebar() -> None:
    st.sidebar.title("DocMind AI")

    if st.sidebar.button("Refresh documents"):
        with st.spinner("Loading documents..."):
            _refresh_documents()

    docs: list[dict[str, Any]] = st.session_state.get("documents", [])

    # Document selection
    if docs:
        current_id = st.session_state.get("active_doc_id")
        doc_by_id = {d.get("doc_id"): d for d in docs}
        id_list = [d.get("doc_id") for d in docs]

        # If nothing selected, pick first.
        if current_id is None:
            first_doc = docs[0]
            _set_active_doc(first_doc)
            current_id = first_doc.get("doc_id")

        active_index = 0
        for i, doc_id in enumerate(id_list):
            if doc_id == current_id:
                active_index = i
                break

        def _fmt(doc_id: str) -> str:
            d = doc_by_id.get(doc_id, {}) or {}
            filename = d.get("filename") or ""
            chunk_count = d.get("chunk_count")
            return f"{filename} ({chunk_count} chunks)"

        selected_id = st.sidebar.selectbox(
            "Active document",
            id_list,
            index=active_index,
            format_func=_fmt,
        )
        _set_active_doc(doc_by_id[selected_id])
    else:
        _set_active_doc(None)
        st.sidebar.caption("No documents yet. Upload a PDF to begin.")

    # Upload
    st.sidebar.divider()
    st.sidebar.subheader("Upload PDF")
    uploaded = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded is not None:
        if st.sidebar.button("Upload to backend", type="primary"):
            with st.spinner("Uploading and indexing..."):
                try:
                    content = uploaded.getvalue()
                    data = _api_upload_pdf(
                        "/upload",
                        filename=uploaded.name or "document.pdf",
                        content=content,
                    )
                    st.sidebar.success(f"Uploaded: {data.get('filename', uploaded.name)}")
                    _refresh_documents()
                except Exception as e:
                    st.sidebar.error(f"Upload failed: {e}")

    active_doc_id = st.session_state.get("active_doc_id")

    # Chunks
    st.sidebar.divider()
    st.sidebar.subheader("Vector chunks")
    if active_doc_id:
        if st.sidebar.button("Load chunks", disabled=st.session_state.get("chunks") is not None):
            with st.spinner("Loading vector chunks..."):
                try:
                    _load_chunks(active_doc_id)
                except Exception as e:
                    st.sidebar.error(f"Failed to load chunks: {e}")

        if st.sidebar.button("Clear chunks", disabled=st.session_state.get("chunks") is None):
            st.session_state["chunks"] = None

        if st.session_state.get("chunks"):
            chunks = st.session_state["chunks"]
            st.sidebar.caption(f"{len(chunks)} chunks loaded")

            # Show a compact preview in the sidebar.
            # Full view is in the main panel below.
            preview = chunks[:3]
            for c in preview:
                st.sidebar.markdown(f"- Chunk {c.get('chunk_index', 0) + 1}")
    else:
        st.sidebar.caption("Select a document first.")

    # Delete
    st.sidebar.divider()
    st.sidebar.subheader("Danger zone")
    if active_doc_id and st.sidebar.button("Delete active document"):
        if st.sidebar.checkbox("I understand this will delete vectors too", key="confirm_delete"):
            with st.spinner("Deleting..."):
                try:
                    _delete_doc(active_doc_id)
                    _set_active_doc(None)
                    st.sidebar.success("Document deleted.")
                except Exception as e:
                    st.sidebar.error(f"Delete failed: {e}")


def _render_main() -> None:
    st.title("DocMind AI")

    # Semantic search (uses /search and the selected doc_id)
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input(
            "Semantic search in active document",
            placeholder="Type a search query...",
            key="search_query",
        )
    with col2:
        do_search = st.button("Search", use_container_width=True)

    active_doc_id = st.session_state.get("active_doc_id")

    if do_search and active_doc_id and search_query.strip():
        with st.spinner("Searching..."):
            try:
                data = _api_get("/search", params={"q": search_query, "doc_id": active_doc_id})
                results = data.get("results", []) or []
                if not results:
                    st.info("No results found.")
                else:
                    for r in results:
                        chunk_index = r.get("chunk_index")
                        score = r.get("score")
                        snippet = r.get("snippet", "")
                        header = f"Chunk {chunk_index + 1}" if isinstance(chunk_index, int) else "Chunk"
                        if score is not None:
                            header += f" • {(score * 100):.0f}%"
                        st.markdown(f"### {header}")
                        st.write(snippet)
            except Exception as e:
                st.error(f"Search failed: {e}")
    elif do_search and not active_doc_id:
        st.info("Select a document first to run semantic search.")

    st.divider()

    # Full chunks view (if loaded)
    chunks = st.session_state.get("chunks")
    if chunks:
        with st.expander("Vector chunks (full text)", expanded=False):
            for c in chunks:
                idx = c.get("chunk_index", 0)
                text = c.get("text", "") or ""
                st.markdown(f"#### Chunk {idx + 1}")
                st.write(text)
                st.divider()

    _render_chat()


def main() -> None:
    st.set_page_config(page_title="DocMind AI", layout="wide")
    _ensure_state()

    # Start backend (optional) before first API calls.
    _start_backend_if_needed()

    # Initial docs load.
    if not st.session_state.get("documents"):
        try:
            with st.spinner("Loading documents..."):
                _refresh_documents()
        except Exception as e:
            st.error(
                "Failed to load documents from backend. "
                "Check that your backend is running and `API_URL` is correct."
                f"\n\nDetails: {e}"
            )

    _render_sidebar()
    _render_main()


if __name__ == "__main__":
    main()

