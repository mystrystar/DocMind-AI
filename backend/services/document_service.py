"""
Document processing service: PDF text extraction and chunking.
Uses pdfplumber for PDF text extraction and tiktoken for token-based chunking
(500 tokens per chunk, 50-token overlap).
"""

import re
import uuid
from pathlib import Path

import pdfplumber
import tiktoken

# Default: 500 tokens per chunk, 50 overlap. For large docs we use bigger chunks to stay within Ollama limits.
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
# Large doc threshold (~200k chars): use 1000-token chunks so 14 MB PDFs produce fewer chunks
LARGE_DOC_TOKEN_THRESHOLD = 50_000
LARGE_CHUNK_SIZE = 1000
LARGE_CHUNK_OVERLAP = 100
# Use cl100k_base (GPT-4/3.5 tokenizer) for consistent token count
ENCODING = tiktoken.get_encoding("cl100k_base")


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract raw text from a PDF file using pdfplumber.
    Falls back to PyPDF2 if pdfplumber yields empty text.
    """
    text_parts = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception:
        # Fallback to PyPDF2
        import PyPDF2
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
    raw = "\n".join(text_parts) if text_parts else ""
    # Normalize whitespace for cleaner chunking
    return re.sub(r"\s+", " ", raw).strip()


def count_tokens(text: str) -> int:
    """Return token count for the given text using cl100k_base encoding (tiktoken)."""
    return len(ENCODING.encode(text))


def chunk_text_by_tokens(
    text: str,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """
    Split text into chunks of roughly `chunk_size` tokens with `overlap` token overlap.
    Overlap helps preserve context at chunk boundaries for better retrieval.
    """
    if not text or not text.strip():
        return []
    tokens = ENCODING.encode(text)
    if len(tokens) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(ENCODING.decode(chunk_tokens))
        if end >= len(tokens):
            break
        start = end - overlap
    return chunks


def process_pdf_to_chunks(file_path: str) -> tuple[str, list[str]]:
    """
    Extract text from PDF and return (normalized_text, list of chunks).
    For very large documents (e.g. 14 MB PDFs), uses larger chunk size to reduce
    total chunks and avoid overloading Ollama embedding.
    """
    raw_text = extract_text_from_pdf(file_path)
    if not raw_text:
        raise ValueError("No text could be extracted from the PDF")
    token_count = count_tokens(raw_text)
    if token_count > LARGE_DOC_TOKEN_THRESHOLD:
        chunks = chunk_text_by_tokens(
            raw_text, LARGE_CHUNK_SIZE, LARGE_CHUNK_OVERLAP
        )
    else:
        chunks = chunk_text_by_tokens(
            raw_text, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS
        )
    return raw_text, chunks


def generate_doc_id() -> str:
    """Generate a unique document ID (UUID4)."""
    return str(uuid.uuid4())
