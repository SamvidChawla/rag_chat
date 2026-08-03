import logging
import time
import random

from google import genai
from google.genai import types
from google.genai.errors import APIError

from app.config import settings

logger = logging.getLogger(__name__)

_client = genai.Client(api_key=settings.gemini_api_key)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _embed_with_retry(
    text: str,
    task_type: str,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> list[float]:
    attempt = 0

    while True:
        try:
            response = _client.models.embed_content(
                model=settings.gemini_embedding_model,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=settings.embedding_dim,
                ),
            )
            return response.embeddings[0].values

        except APIError as e:
            status = getattr(e, "code", None) or getattr(e, "status_code", None)
            attempt += 1

            if status not in RETRYABLE_STATUS_CODES or attempt > max_retries:
                logger.error(
                    "Embedding call failed permanently | status=%s | attempt=%d | task_type=%s",
                    status, attempt, task_type,
                    exc_info=True,
                )
                raise

            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay += random.uniform(0, delay * 0.1)

            logger.warning(
                "Embedding call failed (status=%s), retrying in %.1fs (attempt %d/%d)",
                status, delay, attempt, max_retries,
            )
            time.sleep(delay)

        except Exception:
            logger.error(
                "Embedding call failed with non-retryable error | task_type=%s",
                task_type,
                exc_info=True,
            )
            raise


def embed_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        raise ValueError("embed_batch called with empty text list")

    logger.info("Embedding %d text(s) individually (SDK has no true batch mode)", len(texts))

    vectors = [_embed_with_retry(t, task_type="RETRIEVAL_DOCUMENT") for t in texts]

    logger.info("Embedded %d text(s) successfully (dim=%d)", len(vectors), len(vectors[0]))
    return vectors


def embed_query(text: str) -> list[float]:
    if not text.strip():
        raise ValueError("embed_query called with empty text")

    vector = _embed_with_retry(text, task_type="RETRIEVAL_QUERY")
    logger.info("Query embedded successfully (dim=%d)", len(vector))
    return vector