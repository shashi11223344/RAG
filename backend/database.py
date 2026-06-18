import sqlite3
import json
from pathlib import Path
from typing import Any

DB_PATH = "rag_logs.db"


def init_db() -> None:
    """Create the query_logs table if it doesn't exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_logs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                query           TEXT    NOT NULL,
                answer          TEXT,
                sources         TEXT,
                answer_found    INTEGER NOT NULL DEFAULT 0,
                response_time_ms REAL   NOT NULL,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def log_query(
    query: str,
    answer: str,
    sources: list,
    answer_found: bool,
    response_time_ms: float,
) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO query_logs (query, answer, sources, answer_found, response_time_ms)
            VALUES (?, ?, ?, ?, ?)
            """,
            (query, answer, json.dumps(sources), int(answer_found), response_time_ms),
        )
        conn.commit()


def get_analytics() -> dict[str, Any]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        total_row = conn.execute("SELECT COUNT(*) AS total FROM query_logs").fetchone()
        total = total_row["total"]

        avg_row = conn.execute(
            "SELECT AVG(response_time_ms) AS avg_ms FROM query_logs"
        ).fetchone()
        avg_ms = avg_row["avg_ms"] or 0.0

        frequent = conn.execute(
            """
            SELECT query, COUNT(*) AS count
            FROM query_logs
            GROUP BY query
            ORDER BY count DESC
            LIMIT 10
            """
        ).fetchall()

        unanswered = conn.execute(
            """
            SELECT query, COUNT(*) AS count
            FROM query_logs
            WHERE answer_found = 0
            GROUP BY query
            ORDER BY count DESC
            LIMIT 10
            """
        ).fetchall()

    return {
        "total_queries": total,
        "avg_response_latency_ms": round(avg_ms, 2),
        "most_frequent_queries": [dict(r) for r in frequent],
        "unanswered_queries": [dict(r) for r in unanswered],
    }
