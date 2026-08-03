import logging
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

logger = logging.getLogger(__name__)

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_text(text: str, source_name: str, extra_metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """
    Split raw document text into overlapping chunks.

    Returns:
        list of {"content": str, "source_name": str, "metadata": dict}
    """
    if not text.strip():
        logger.error("chunk_text called with empty text for source: %s", source_name)
        raise ValueError(f"Cannot chunk empty text: {source_name}")

    pieces = _splitter.split_text(text)

    if len(pieces) <= 1:
        logger.warning(
            "Document produced only %d chunk(s) — chunk_size (%d) may be larger than document, or splitter found no natural breakpoints: %s",
            len(pieces), settings.chunk_size, source_name,
        )

    base_metadata = extra_metadata or {}

    chunks = [
        {
            "content": piece,
            "source_name": source_name,
            "metadata": {**base_metadata, "chunk_index": i, "total_chunks": len(pieces)},
        }
        for i, piece in enumerate(pieces)
    ]

    logger.info("Chunked '%s' into %d chunks (chunk_size=%d, overlap=%d)",
                source_name, len(chunks), settings.chunk_size, settings.chunk_overlap)

    return chunks