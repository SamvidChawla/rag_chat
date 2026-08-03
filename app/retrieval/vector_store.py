import json
from contextlib import contextmanager
from typing import Any

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool

from app.config import settings

_pool = ThreadedConnectionPool(
    minconn=settings.db_pool_min,
    maxconn=settings.db_pool_max,
    dsn=settings.database_url,
)


@contextmanager
def get_connection():
    conn = _pool.getconn()
    try:
        yield conn
    finally:
        _pool.putconn(conn)


def close_pool() -> None:
    _pool.closeall()


def insert_chunks(chunks: list[dict[str, Any]]) -> None:
    """
    chunks: list of dicts with keys:
        source_name (str), content (str), metadata (dict), embedding (list[float])
    """
    if not chunks:
        return

    query = """
        INSERT INTO documents (source_name, content, metadata, embedding)
        VALUES %s
    """
    values = [
        (
            c["source_name"],
            c["content"],
            json.dumps(c.get("metadata", {})),
            c["embedding"],
        )
        for c in chunks
    ]

    with get_connection() as conn:
        try:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    query,
                    values,
                    template="(%s, %s, %s, %s::vector)",
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def search(query_embedding: list[float], top_k: int | None = None) -> list[dict[str, Any]]:
    top_k = top_k or settings.top_k
    dim = settings.embedding_dim

    query = f"""
        SELECT id, source_name, content, metadata,
               embedding::halfvec({dim}) <=> %s::halfvec({dim}) AS distance
        FROM documents
        ORDER BY embedding::halfvec({dim}) <=> %s::halfvec({dim})
        LIMIT %s
    """

    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, (query_embedding, query_embedding, top_k))
            return cur.fetchall()