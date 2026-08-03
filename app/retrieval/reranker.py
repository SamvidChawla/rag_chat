import logging
import time
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

_model = None  # lazy-loaded, only if reranking is enabled


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading reranker model: %s", settings.reranker_model)
        start = time.time()
        _model = CrossEncoder(settings.reranker_model)
        logger.info("Reranker model loaded in %.2fs", time.time() - start)
    return _model


def rerank(query: str, chunks: list[dict[str, Any]], top_n: int | None = None) -> list[dict[str, Any]]:
    top_n = top_n or settings.rerank_top_k

    if not settings.enable_reranking:
        return chunks[:top_n]

    if not chunks:
        return chunks

    model = _get_model()
    pairs = [(query, c["content"]) for c in chunks]

    logger.info("Reranking %d chunks for query", len(chunks))
    scores = model.predict(pairs)

    scored = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    top_chunks = [c for c, _ in scored[:top_n]]

    logger.info(
        "Reranked: top score=%.4f, bottom kept score=%.4f",
        scored[0][1], scored[min(top_n, len(scored)) - 1][1],
    )

    return top_chunks