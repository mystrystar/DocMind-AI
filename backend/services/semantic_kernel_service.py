"""
Semantic Kernel service: RAG with OpenAI GPT-4o.
Creates a Kernel with OpenAIChatCompletion, and a RAG prompt that instructs the model
to answer ONLY from the provided context and say "I don't know" when context is insufficient.
"""

import os
from typing import List

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.functions import KernelArguments

# Model for chat (GPT-4o as specified)
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL_ID", "gpt-4o")

# RAG prompt template: {{$context}} and {{$question}} are filled via KernelArguments
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


def create_kernel() -> Kernel:
    """
    Create and return a Semantic Kernel Kernel with OpenAI chat completion service.
    Uses OPENAI_API_KEY from environment; model is gpt-4o by default.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    kernel = Kernel()
    chat_service = OpenAIChatCompletion(
        ai_model_id=CHAT_MODEL,
        api_key=api_key,
    )
    kernel.add_service(chat_service)
    return kernel


async def get_rag_answer(
    context_chunks: List[str],
    question: str,
    kernel: Kernel | None = None,
) -> str:
    """
    Run RAG: combine chunks into context, call Semantic Kernel invoke_prompt with
    KernelArguments(context=..., question=...), and return the model's answer.
    If context is empty, returns a fallback without calling the model.
    """
    if not context_chunks:
        return "I don't know. No relevant passages were found in the document for your question."
    context = "\n\n---\n\n".join(
        f"[Chunk {i + 1}]\n{c}" for i, c in enumerate(context_chunks)
    )
    if kernel is None:
        kernel = create_kernel()
    arguments = KernelArguments(context=context, question=question)
    result = await kernel.invoke_prompt(
        RAG_PROMPT_TEMPLATE,
        arguments=arguments,
    )
    if not result:
        return "I don't know."
    # FunctionResult: use get_inner_content() for KernelContent or value
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
