"""SQLite logging for every query result produced by the RAG pipeline.

Creates eval/logs.db and the query_logs table on first use.
"""

import logging
import sqlite3
from pathlib import Path

from nba_types import SQLITE_DB_PATH, QueryResult

logger = logging.getLogger(__name__)


def _get_connection() -> sqlite3.Connection:
    """Open (or create) the SQLite database and ensure the schema exists."""
    db_path = Path(SQLITE_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            config_name TEXT NOT NULL,
            k INTEGER NOT NULL,
            chunk_size INTEGER NOT NULL,
            chunk_overlap INTEGER NOT NULL,
            faithfulness REAL NOT NULL,
            answer_relevance REAL NOT NULL,
            passed INTEGER NOT NULL,
            retry_count INTEGER NOT NULL,
            low_confidence INTEGER NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def log_result(result: QueryResult) -> None:
    """Insert a QueryResult into the query_logs table.

    Args:
        result: The completed QueryResult to persist.
    """
    try:
        conn = _get_connection()
        conn.execute(
            """
            INSERT INTO query_logs (
                question, answer, config_name, k, chunk_size, chunk_overlap,
                faithfulness, answer_relevance, passed,
                retry_count, low_confidence, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.question,
                result.answer,
                result.config_used.name,
                result.config_used.k,
                result.config_used.chunk_size,
                result.config_used.chunk_overlap,
                result.score.faithfulness,
                result.score.answer_relevance,
                int(result.score.passed),
                result.retry_count,
                int(result.low_confidence),
                result.timestamp.isoformat(),
            ),
        )
        conn.commit()
        conn.close()
        logger.info("Logged query to SQLite: '%s...'", result.question[:60])
    except sqlite3.Error as exc:
        logger.error("Failed to log query result: %s", exc)


def fetch_logs(limit: int | None = None) -> list[dict]:
    """Retrieve query logs from SQLite, newest first.

    Args:
        limit: Maximum number of rows to return. None returns all rows.

    Returns:
        List of dicts with one key per column.
    """
    try:
        conn = _get_connection()
        query = "SELECT * FROM query_logs ORDER BY id DESC"
        if limit is not None:
            query += f" LIMIT {limit}"
        cursor = conn.execute(query)
        columns = [desc[0] for desc in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return rows
    except sqlite3.Error as exc:
        logger.error("Failed to fetch logs: %s", exc)
        return []
