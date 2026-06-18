import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend import database, rag
from backend.models import (
    AnalyticsResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
)

app = FastAPI(
    title="AWS Customer Agreement RAG API",
    description="RAG-powered Q&A over the AWS Customer Agreement with usage analytics.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    database.init_db()
    loaded = rag.load_index()
    if loaded:
        print("Loaded existing FAISS index from disk.")
    else:
        print("No index found. Call POST /ingest to process the document.")


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: Optional[IngestRequest] = Body(default=None)) -> IngestResponse:
    """Parse the PDF, generate embeddings and persist the FAISS index."""
    if request is None:
        request = IngestRequest()
    pdf_path = Path(request.pdf_path)
    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"PDF not found at '{pdf_path}'. Place the file there and retry.",
        )
    try:
        n_chunks = rag.ingest(str(pdf_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return IngestResponse(
        message=f"Document ingested successfully from '{pdf_path}'.",
        chunks_created=n_chunks,
    )


@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest) -> QueryResponse:
    """Answer a question using the RAG pipeline and log the interaction."""
    if not rag.is_index_loaded():
        raise HTTPException(
            status_code=400,
            detail="No document ingested yet. Call POST /ingest first.",
        )

    t0 = time.perf_counter()
    try:
        answer, sources, answer_found = rag.ask(request.query)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {exc}") from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000

    database.log_query(
        query=request.query,
        answer=answer,
        sources=sources,
        answer_found=answer_found,
        response_time_ms=elapsed_ms,
    )

    return QueryResponse(
        query=request.query,
        answer=answer,
        sources=[SourceChunk(**s) for s in sources],
        answer_found=answer_found,
        response_time_ms=round(elapsed_ms, 2),
    )


@app.get("/analytics", response_model=AnalyticsResponse)
def analytics() -> AnalyticsResponse:
    """Return usage analytics backed by SQL aggregation queries."""
    data = database.get_analytics()
    return AnalyticsResponse(**data)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "index_loaded": rag.is_index_loaded()}
