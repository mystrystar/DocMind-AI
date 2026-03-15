"""
ChromaDB vector service: store document chunks and run semantic search.
Uses Ollama for embeddings only (model: nomic-embed-text) at http://localhost:11434.
No OpenAI or API keys required.
"""

import os

import chromadb
from chromadb.config import Settings

# Ollama runs locally
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_data")


def _get_ollama_embedding(text: str) -> list[float]:
    """Single text embedding via Ollama nomic-embed-text."""
    import ollama
    # ollama client uses OLLAMA_HOST env by default
    r = ollama.embed(model=OLLAMA_EMBEDDING_MODEL, input=text)
    if not r or not r.embeddings:
        raise RuntimeError("Ollama embed returned no embeddings")
    return r.embeddings[0]


# Small batch size for Ollama embed to avoid timeouts/OOM on large docs (e.g. 14 MB PDFs)
OLLAMA_EMBED_BATCH_SIZE = int(os.getenv("OLLAMA_EMBED_BATCH_SIZE", "25"))


def _get_ollama_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Batch embeddings via Ollama in small chunks so large documents (e.g. 14 MB) can be processed."""
    if not texts:
        return []
    import ollama
    all_embeddings = []
    for i in range(0, len(texts), OLLAMA_EMBED_BATCH_SIZE):
        batch = texts[i : i + OLLAMA_EMBED_BATCH_SIZE]
        r = ollama.embed(model=OLLAMA_EMBEDDING_MODEL, input=batch)
        if not r or not r.embeddings:
            raise RuntimeError("Ollama embed returned no embeddings")
        all_embeddings.extend(r.embeddings)
    return all_embeddings


class VectorService:
    """
    ChromaDB-backed vector store: one collection per document (doc_id).
    All embeddings from Ollama (nomic-embed-text).
    """

    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )

    def _collection_name(self, doc_id: str) -> str:
        safe = "".join(c if c.isalnum() or c == "-" else "_" for c in doc_id)
        return f"doc_{safe}" if not safe.startswith("doc_") else safe

    def add_document_chunks(self, doc_id: str, chunks: list[str]) -> None:
        if not chunks:
            return
        name = self._collection_name(doc_id)
        try:
            self._client.delete_collection(name)
        except Exception:
            pass
        collection = self._client.create_collection(
            name=name,
            metadata={"doc_id": doc_id},
        )
        # Embed in batches (handles large docs e.g. 14 MB PDFs) then add to Chroma in chunks
        all_embeddings = _get_ollama_embeddings_batch(chunks)
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metadatas = [{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]
        # Add to Chroma in batches of 100 to avoid huge single request
        add_batch_size = 100
        for j in range(0, len(chunks), add_batch_size):
            end = min(j + add_batch_size, len(chunks))
            collection.add(
                ids=ids[j:end],
                documents=chunks[j:end],
                embeddings=all_embeddings[j:end],
                metadatas=metadatas[j:end],
            )

    def search(
        self,
        doc_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[int, float, str]]:
        name = self._collection_name(doc_id)
        try:
            collection = self._client.get_collection(name=name)
        except Exception:
            return []
        query_embedding = _get_ollama_embedding(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        if not results or not results["ids"] or not results["ids"][0]:
            return []
        out = []
        distances = results["distances"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        for i, meta in enumerate(metadatas):
            chunk_index = int(meta.get("chunk_index", i))
            dist = distances[i] if i < len(distances) else 0
            score = 1.0 / (1.0 + float(dist))
            text = documents[i] if i < len(documents) else ""
            out.append((chunk_index, score, text))
        return out

    def get_document_chunks(self, doc_id: str) -> list[tuple[int, str]]:
        """Return all stored chunks for a document as (chunk_index, text)."""
        name = self._collection_name(doc_id)
        try:
            collection = self._client.get_collection(name=name)
        except Exception:
            return []
        n = collection.count()
        if n == 0:
            return []
        # get() by metadata filter or by known ids
        try:
            data = collection.get(
                where={"doc_id": doc_id},
                include=["documents", "metadatas"],
                limit=n,
            )
        except Exception:
            data = None
        if not data or not data["ids"]:
            # Fallback: fetch by ids (we use doc_id_0, doc_id_1, ...)
            try:
                ids = [f"{doc_id}_{i}" for i in range(n)]
                data = collection.get(ids=ids, include=["documents", "metadatas"])
            except Exception:
                return []
        if not data or not data["ids"]:
            return []
        docs = data["documents"]
        metas = data["metadatas"]
        out = []
        for i, meta in enumerate(metas):
            idx = int(meta.get("chunk_index", i))
            text = docs[i] if i < len(docs) else ""
            out.append((idx, text))
        out.sort(key=lambda x: x[0])
        return out

    def document_exists(self, doc_id: str) -> bool:
        name = self._collection_name(doc_id)
        try:
            self._client.get_collection(name=name)
            return True
        except Exception:
            return False

    def delete_document(self, doc_id: str) -> None:
        name = self._collection_name(doc_id)
        try:
            self._client.delete_collection(name)
        except Exception:
            pass
