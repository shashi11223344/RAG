# AWS Customer Agreement — RAG Q&A System

A Retrieval-Augmented Generation (RAG) system that answers questions about the AWS Customer Agreement, backed by a FastAPI API with SQLite usage logging and a Streamlit analytics dashboard.

---

## Architecture

```
┌──────────────────────┐        HTTP        ┌────────────────────────────┐
│  Streamlit Frontend  │ ─────────────────▶ │   FastAPI Backend          │
│  (port 8501)         │                    │   (port 8000)              │
└──────────────────────┘                    │                            │
                                            │  POST /ingest              │
                                            │    └─ PyMuPDF → chunks     │
                                            │    └─ sentence-transformers│
                                            │    └─ FAISS IndexFlatIP    │
                                            │                            │
                                            │  POST /ask                 │
                                            │    └─ FAISS retrieval (k=5)│
                                            │    └─ Claude Haiku (LLM)   │
                                            │    └─ SQLite log_query()   │
                                            │                            │
                                            │  GET /analytics            │
                                            │    └─ SQL GROUP BY/AVG     │
                                            └────────────────────────────┘
```

---

## Key Design Decisions

### Chunking Strategy
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Chunk size | 800 characters | Captures a complete legal sub-clause (typically 100–200 words). Smaller chunks fragment clauses; larger ones add irrelevant context. |
| Overlap | 100 characters | Prevents clause-boundary sentences from being split across adjacent chunks. |

### Retrieval
- **top-k = 5**: Legal docs cross-reference many sections. Five chunks provide sufficient context for compound questions while staying well within the LLM's context window.
- **Similarity metric**: Cosine similarity (inner-product after L2 normalisation), which is more stable than raw L2 distance for sentence embeddings.

### Embeddings
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` — free, runs locally, 22M parameters, good semantic quality for retrieval tasks.

### LLM
- **Provider**: Groq API — `llama-3.3-70b-versatile`. Groq's inference is extremely fast (low TTFT), and the model is free-tier compatible. Prompt explicitly instructs the model to say it cannot find information if the answer is absent, preventing hallucination.

### SQL Schema
```sql
CREATE TABLE query_logs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    query            TEXT    NOT NULL,
    answer           TEXT,
    sources          TEXT,           -- JSON array of {chunk_id, text}
    answer_found     INTEGER NOT NULL DEFAULT 0,
    response_time_ms REAL    NOT NULL,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## Setup & Run

### Prerequisites
- Python 3.10+
- A Groq API key (free at console.groq.com)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your API key
```bash
# Windows PowerShell
$env:GROQ_API_KEY = "gsk_..."

# macOS / Linux
export GROQ_API_KEY="gsk_..."
```

Or copy `.env.example` to `.env` and fill in the key.

### 3. Start the FastAPI backend
```bash
uvicorn backend.main:app --reload --port 8000
```

### 4. Start the Streamlit frontend (separate terminal)
```bash
streamlit run frontend/app.py --server.port 8501
```

### 5. Ingest the document
Open the Streamlit app at `http://localhost:8501`, go to the **Setup** tab, and click **Ingest Document** — or call the API directly:
```bash
curl -X POST http://localhost:8000/ingest \
     -H "Content-Type: application/json" \
     -d '{"pdf_path": "data/aws_customer_agreement.pdf"}'
```

### 6. Seed test queries (optional but recommended for analytics)
```bash
python seed_queries.py
```
This fires 35 pre-written queries (25 answerable + 10 out-of-scope) at `/ask` to populate the logs table with realistic data.

---

## API Reference

### `POST /ingest`
Parse and embed the PDF.
```json
{ "pdf_path": "data/aws_customer_agreement.pdf" }
```

### `POST /ask`
Ask a question.
```json
{ "query": "What is the liability cap under the AWS Customer Agreement?" }
```
Response:
```json
{
  "query": "...",
  "answer": "...",
  "sources": [{ "chunk_id": 42, "text": "..." }],
  "answer_found": true,
  "response_time_ms": 823.4
}
```

### `GET /analytics`
Returns SQL-aggregated usage data: most frequent queries, unanswered queries, average response latency.

### `GET /health`
Quick health check including whether the vector index is loaded.

---

## Project Structure
```
.
├── backend/
│   ├── __init__.py
│   ├── main.py        # FastAPI app & route handlers
│   ├── rag.py         # PDF parsing, chunking, embedding, retrieval, LLM call
│   ├── database.py    # SQLite init, log_query(), get_analytics()
│   └── models.py      # Pydantic request/response models
├── frontend/
│   └── app.py         # Streamlit UI (chat + analytics tabs)
├── data/
│   └── aws_customer_agreement.pdf
├── seed_queries.py    # Populates logs with 35 test queries
├── requirements.txt
├── .env.example
└── README.md
```
