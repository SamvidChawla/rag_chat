import logging
from pathlib import Path
from typing import Any

from app.ingestion.loader import load_document, SUPPORTED_EXTENSIONS
from app.ingestion.chunker import chunk_text
from app.ingestion.embedder import embed_batch
from app.retrieval.vector_store import insert_chunks

logger = logging.getLogger(__name__)


def ingest_document(file_path: str) -> dict[str, Any]:
    """
    Runs the full pipeline for a single file: load -> chunk -> embed -> store.
    Returns stats about the ingestion. Raises on failure (caller decides how to handle).
    """
    logger.info("Starting ingestion: %s", file_path)

    loaded = load_document(file_path)
    chunks = chunk_text(
        text=loaded["text"],
        source_name=loaded["metadata"]["source_name"],
        extra_metadata=loaded["metadata"],
    )

    contents = [c["content"] for c in chunks]
    embeddings = embed_batch(contents)

    if len(embeddings) != len(chunks):
        logger.error(
            "Embedding count mismatch: %d chunks vs %d embeddings for %s",
            len(chunks), len(embeddings), file_path,
        )
        raise RuntimeError("Embedding count does not match chunk count")

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    insert_chunks(chunks)

    logger.info("Ingestion complete: %s (%d chunks stored)", file_path, len(chunks))

    return {
        "source_name": loaded["metadata"]["source_name"],
        "chunks_stored": len(chunks),
    }