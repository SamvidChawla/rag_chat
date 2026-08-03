import logging

from google import genai
from google.genai import types

from app.config import settings

logger = logging.getLogger(__name__)

_client = genai.Client(api_key=settings.gemini_api_key)


def _embed(
    texts: list[str],
    task_type: str,
) -> list[list[float]]:
    if not texts:
        raise ValueError("embed called with empty text list")

    logger.debug(
        "Embedding %d text(s), task_type=%s, dim=%d",
        len(texts),
        task_type,
        settings.embedding_dim,
    )

    try:
        response = _client.models.embed_content(
            model=settings.gemini_embedding_model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=settings.embedding_dim,
            ),
        )
    except Exception:
        logger.error(
            "Embedding call failed | model=%s | task_type=%s | batch_size=%d",
            settings.gemini_embedding_model,
            task_type,
            len(texts),
            exc_info=True,
        )
        raise

    vectors = [e.values for e in response.embeddings]
    logger.info("Embedded %d text(s) successfully (dim=%d)", len(vectors), len(vectors[0]))
    return vectors


def embed_batch(texts: list[str]) -> list[list[float]]:
    """For ingestion: embed document chunks."""
    return _embed(texts, task_type="RETRIEVAL_DOCUMENT")


def embed_query(text: str) -> list[float]:
    """For search: embed a single user query."""
    return _embed([text], task_type="RETRIEVAL_QUERY")[0]