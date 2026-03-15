"""
ChromaDB vector service: store document chunks and run semantic search.
Supports two modes:
- Local (free): ChromaDB's DefaultEmbeddingFunction (all-MiniLM-L6-v2, runs on your machine).
- OpenAI: uses OpenAI embeddings (requires API key and quota).
Set USE_LOCAL_EMBEDDINGS=true in .env for free tier (no OpenAI needed for embeddings).
"""

import os
from typing import Optional

import chromadb
from chromadb.config import Settings

# Persist ChromaDB under ./chroma_data relative to backend
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_data")

# Set to "true" or "1" to use local embeddings (free, no OpenAI)
USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "").strip().lower() in ("true", "1", "yes")


def _get_local_embedding_function():
    """Local embeddings: all-MiniLM-L6-v2, runs on your machine. No API key."""
    try:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        return DefaultEmbeddingFunction()
    except ImportError:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


# --- OpenAI path (when not using local embeddings) ---
if not USE_LOCAL_EMBEDDINGS:
    from openai import OpenAI
    EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL_ID", "text-embedding-3-small")

    def _get_openai_client() -> OpenAI:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when not using local embeddings. Set USE_LOCAL_EMBEDDINGS=true for free tier.")
        return OpenAI(api_key=api_key)

    def _get_embedding(text: str, client: Optional[OpenAI] = None) -> list[float]:
        client = client or _get_openai_client()
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        return resp.data[0].embedding

    def _get_embeddings_batch(texts: list[str], client: Optional[OpenAI] = None) -> list[list[float]]:
        client = client or _get_openai_client()
        batch_size = 100
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            sorted_data = sorted(resp.data, key=lambda x: x.index)
            all_embeddings.extend([d.embedding for d in sorted_data])
        return all_embeddings


class VectorService:
    """
    ChromaDB-backed vector store: one collection per document (doc_id).
    Uses local embeddings (free) when USE_LOCAL_EMBEDDINGS=true, else OpenAI.
    """

    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self._use_local = USE_LOCAL_EMBEDDINGS
        self._openai = None
        self._local_ef = None

    def _openai_client(self):
        if not USE_LOCAL_EMBEDDINGS and self._openai is None:
            self._openai = _get_openai_client()
        return self._openai

    def _local_embedding_fn(self):
        if self._local_ef is None:
            self._local_ef = _get_local_embedding_function()
        return self._local_ef

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

        if self._use_local:
            # Free tier: Chroma embeds with DefaultEmbeddingFunction (all-MiniLM-L6-v2)
            collection = self._client.create_collection(
                name=name,
                metadata={"doc_id": doc_id},
                embedding_function=self._local_embedding_fn(),
            )
            ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
            metadatas = [{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]
            collection.add(ids=ids, documents=chunks, metadatas=metadatas)
        else:
            collection = self._client.create_collection(
                name=name,
                metadata={"doc_id": doc_id},
            )
            ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
            metadatas = [{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]
            embeddings = _get_embeddings_batch(chunks, self._openai_client())
            collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
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

        if self._use_local:
            results = collection.query(
                query_texts=[query],
                n_results=min(top_k, collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        else:
            query_embedding = _get_embedding(query, self._openai_client())
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
