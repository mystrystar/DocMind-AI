"""
Chat router: POST /chat — retrieve top 5 chunks from ChromaDB, pass to Semantic Kernel, return answer with citations.
"""

from fastapi import APIRouter, HTTPException

from models.schemas import ChatRequest, ChatResponse, Citation
from services.vector_service import VectorService
from services.semantic_kernel_service import get_rag_answer, create_kernel
from services.document_store import get_document

router = APIRouter(prefix="/chat", tags=["chat"])
_vector_service: VectorService | None = None
_kernel = None


def get_vector_service() -> VectorService:
    global _vector_service
    if _vector_service is None:
        _vector_service = VectorService()
    return _vector_service


def get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = create_kernel()
    return _kernel


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Take user question + document_id → retrieve top 5 relevant chunks from ChromaDB
    → pass as context to Semantic Kernel → return grounded answer with source citations.
    """
    doc_id = request.document_id
    if get_document(doc_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    vector_service = get_vector_service()
    results = vector_service.search(doc_id=doc_id, query=request.question, top_k=5)
    chunks = [r[2] for r in results]
    scores = [r[1] for r in results]
    indices = [r[0] for r in results]
    answer = await get_rag_answer(context_chunks=chunks, question=request.question, kernel=get_kernel())
    citations = [
        Citation(chunk_index=idx, snippet=snippet[:500] + ("..." if len(snippet) > 500 else ""), score=score)
        for idx, score, snippet in zip(indices, scores, chunks)
    ]
    return ChatResponse(
        answer=answer,
        sources=citations,
        document_id=doc_id,
    )
