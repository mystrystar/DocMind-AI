"""
RAG chat service: answer from context using either OpenAI (GPT-4o) or Ollama (free, local).
Set OLLAMA_BASE_URL=http://localhost:11434 to use Ollama instead of OpenAI (no API key needed).
"""

import os
from typing import List

# RAG prompt template
RAG_PROMPT_TEMPLATE = """You are a helpful document Q&A assistant. You must follow these rules strictly:

1. Answer ONLY using the provided context below. Do not use external knowledge.
2. If the context does not contain enough information to answer the question, respond with exactly: "I don't know" or "The provided context does not contain enough information to answer that."
3. Do not make up or assume facts. Cite which part of the context supports your answer when possible.
4. Keep answers concise and grounded in the context.

Context from the document:
---
{{context}}
---

User question: {{question}}"""


def _use_ollama() -> bool:
    return bool(os.getenv("OLLAMA_BASE_URL", "").strip())


def _build_context(context_chunks: List[str]) -> str:
    return "\n\n---\n\n".join(
        f"[Chunk {i + 1}]\n{c}" for i, c in enumerate(context_chunks)
    )


async def get_rag_answer(
    context_chunks: List[str],
    question: str,
    kernel=None,
) -> str:
    """
    Run RAG: combine chunks into context, then call either Ollama (free) or OpenAI via Semantic Kernel.
    """
    if not context_chunks:
        return "I don't know. No relevant passages were found in the document for your question."

    context = _build_context(context_chunks)
    prompt = RAG_PROMPT_TEMPLATE.replace("{{context}}", context).replace("{{question}}", question)

    if _use_ollama():
        return await _get_rag_answer_ollama(prompt)
    return await _get_rag_answer_openai(prompt, kernel)


async def _get_rag_answer_ollama(prompt: str) -> str:
    """Call local Ollama (free). Requires Ollama running with a model e.g. llama3.1."""
    from openai import OpenAI
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/v1"
    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    client = OpenAI(base_url=base_url, api_key="ollama")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
        )
        if resp.choices and resp.choices[0].message.content:
            return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"I couldn't get an answer from the local model. Is Ollama running? (Error: {e})"
    return "I don't know."


async def _get_rag_answer_openai(prompt: str, kernel=None) -> str:
    """Call OpenAI via Semantic Kernel. Uses prompt built by caller (full string)."""
    from semantic_kernel import Kernel
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
    from semantic_kernel.functions import KernelArguments

    CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL_ID", "gpt-4o")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OpenAI API key is not set. Set OPENAI_API_KEY in .env or use Ollama (OLLAMA_BASE_URL=http://localhost:11434) for free chat."

    if kernel is None:
        kernel = Kernel()
        kernel.add_service(OpenAIChatCompletion(ai_model_id=CHAT_MODEL, api_key=api_key))

    # Single user message with full RAG prompt (context + question already in prompt)
    SK_TEMPLATE = """You are a helpful document Q&A assistant. You must follow these rules strictly:

1. Answer ONLY using the provided context below. Do not use external knowledge.
2. If the context does not contain enough information to answer the question, respond with exactly: "I don't know" or "The provided context does not contain enough information to answer that."
3. Do not make up or assume facts. Cite which part of the context supports your answer when possible.
4. Keep answers concise and grounded in the context.

{{$prompt}}"""
    arguments = KernelArguments(prompt=prompt)
    result = await kernel.invoke_prompt(SK_TEMPLATE, arguments=arguments)
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


def create_kernel():
    """Create Semantic Kernel (only used when not using Ollama)."""
    if _use_ollama():
        return None
    from semantic_kernel import Kernel
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    kernel = Kernel()
    kernel.add_service(OpenAIChatCompletion(
        ai_model_id=os.getenv("OPENAI_CHAT_MODEL_ID", "gpt-4o"),
        api_key=api_key,
    ))
    return kernel
