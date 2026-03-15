"""
RAG chat service: answer from context using Ollama via Semantic Kernel (OllamaChatCompletion).
Uses llama3.2 or mistral (configurable). Ollama at http://localhost:11434. No OpenAI or API keys.
"""

import os
from typing import List

from semantic_kernel import Kernel
from semantic_kernel.functions import KernelArguments

# Ollama chat model (must be pulled: ollama pull llama3.2 / ollama pull mistral)
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

# RAG prompt template: {{$context}} and {{$question}} filled via KernelArguments
RAG_PROMPT_TEMPLATE = """You are a helpful document Q&A assistant. You must follow these rules strictly:

1. Answer ONLY using the provided context below. Do not use external knowledge.
2. If the context does not contain enough information to answer the question, respond with exactly: "I don't know" or "The provided context does not contain enough information to answer that."
3. Do not make up or assume facts. Cite which part of the context supports your answer when possible.
4. Keep answers concise and grounded in the context.

Context from the document:
---
{{$context}}
---

User question: {{$question}}"""


def _build_context(context_chunks: List[str]) -> str:
    return "\n\n---\n\n".join(
        f"[Chunk {i + 1}]\n{c}" for i, c in enumerate(context_chunks)
    )


def create_kernel() -> Kernel:
    """Create Semantic Kernel with OllamaChatCompletion (no API key)."""
    try:
        from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion
    except ImportError:
        raise ImportError(
            "Ollama connector not installed. Run: pip install 'semantic-kernel[ollama]'"
        )
    kernel = Kernel()
    # host = Ollama server URL (e.g. http://localhost:11434)
    service = OllamaChatCompletion(
        ai_model_id=OLLAMA_CHAT_MODEL,
        host=OLLAMA_BASE_URL,
    )
    kernel.add_service(service)
    return kernel


async def get_rag_answer(
    context_chunks: List[str],
    question: str,
    kernel: Kernel | None = None,
) -> str:
    """
    Run RAG: combine chunks into context, call Ollama via Semantic Kernel, return answer.
    """
    if not context_chunks:
        return "I don't know. No relevant passages were found in the document for your question."

    context = _build_context(context_chunks)
    if kernel is None:
        kernel = create_kernel()
    arguments = KernelArguments(context=context, question=question)
    try:
        result = await kernel.invoke_prompt(RAG_PROMPT_TEMPLATE, arguments=arguments)
    except Exception as e:
        return f"I couldn't get an answer from Ollama. Is it running? (Error: {e})"

    if not result:
        return "I don't know."
    try:
        inner = result.get_inner_content() if hasattr(result, "get_inner_content") else result.value
        if inner is not None and hasattr(inner, "content"):
            return inner.content or "I don't know."
        if inner is not None:
            return str(inner)
    except Exception:
        pass
    if hasattr(result, "value") and result.value:
        return str(result.value)
    return "I don't know."
