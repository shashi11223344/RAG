"""
RAG pipeline for the AWS Customer Agreement Q&A system.

Chunking strategy:
  - Chunk size: 800 characters
  - Overlap:    100 characters
  Rationale: The AWS Customer Agreement is dense legal prose where a single
  sub-clause typically spans 100-200 words (~600-1200 chars). 800 chars
  captures a complete sub-clause in most cases. 100-char overlap prevents
  boundary sentences from being split across adjacent chunks, ensuring
  retrieval doesn't miss a clause that straddles a chunk boundary.

Retrieval: top-k = 5
  Legal documents cross-reference many sections. Fetching 5 chunks gives
  enough context to answer compound questions (e.g. termination + fees)
  while keeping the prompt focused and under model context limits.
"""

import os
import pickle
from pathlib import Path

import faiss
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.pkl"

NOT_FOUND_PHRASE = "cannot find information about this in the AWS Customer Agreement"

_encoder: SentenceTransformer | None = None
_index: faiss.Index | None = None
_chunks: list[str] = []


def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(EMBEDDING_MODEL)
    return _encoder


def is_index_loaded() -> bool:
    return _index is not None and len(_chunks) > 0


def load_index() -> bool:
    """Load persisted FAISS index and chunks from disk. Returns True if found."""
    global _index, _chunks
    if Path(INDEX_FILE).exists() and Path(CHUNKS_FILE).exists():
        _index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE, "rb") as f:
            _chunks = pickle.load(f)
        return True
    return False


def _extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    return "\n".join(pages)


def _chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def ingest(pdf_path: str) -> int:
    """Parse PDF, embed chunks, persist index. Returns number of chunks created."""
    global _index, _chunks
    text = _extract_text(pdf_path)
    _chunks = _chunk_text(text)

    encoder = _get_encoder()
    embeddings = encoder.encode(_chunks, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    _index = faiss.IndexFlatIP(dim)  # inner-product after L2 norm == cosine similarity
    _index.add(embeddings)

    faiss.write_index(_index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(_chunks, f)

    return len(_chunks)


def _retrieve(query: str) -> list[tuple[str, int]]:
    encoder = _get_encoder()
    q_emb = encoder.encode([query])
    q_emb = np.array(q_emb, dtype=np.float32)
    faiss.normalize_L2(q_emb)

    _, indices = _index.search(q_emb, TOP_K)
    return [(_chunks[i], int(i)) for i in indices[0] if 0 <= i < len(_chunks)]


def _build_prompt(query: str, context: str) -> str:
    return f"""You are a precise assistant that answers questions exclusively based on the AWS Customer Agreement document.

Instructions:
- Answer using ONLY the provided context passages.
- If the answer is not present in the context, respond with exactly: "I {NOT_FOUND_PHRASE}."
- Quote relevant clauses when helpful. Be concise but complete.

Context from AWS Customer Agreement:
{context}

Question: {query}

Answer:"""


def ask(query: str) -> tuple[str, list[dict], bool]:
    """
    Run the full RAG pipeline for a user query.
    Returns (answer, sources, answer_found).
    """
    if not is_index_loaded():
        raise RuntimeError("No document ingested yet. Call POST /ingest first.")

    context_chunks = _retrieve(query)
    context = "\n\n---\n\n".join(chunk for chunk, _ in context_chunks)

    prompt = _build_prompt(query, context)

    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    answer: str = response.choices[0].message.content.strip()
    answer_found = NOT_FOUND_PHRASE not in answer.lower()

    sources = [
        {"chunk_id": idx, "text": chunk[:300] + ("..." if len(chunk) > 300 else "")}
        for chunk, idx in context_chunks
    ]

    return answer, sources, answer_found
