import logging

from google import genai
from google.genai import types

from app.config import settings

logger = logging.getLogger(__name__)

_client = genai.Client(api_key=settings.gemini_api_key)

SYSTEM_INSTRUCTION = (
    "You are an enterprise assistant. Answer the user's question using ONLY the "
    "provided context chunks. Cite the source_name for any claim you make. "
    "If the context does not contain enough information to answer, say so explicitly "
    "instead of guessing."
)


def _build_prompt(query: str, context_chunks: list[dict]) -> str:
    context_block = "\n\n".join(
        f"[Source: {c['source_name']}]\n{c['content']}" for c in context_chunks
    )
    return f"Context:\n{context_block}\n\nQuestion: {query}"


def generate_answer(query: str, context_chunks: list[dict]) -> str:
    if not context_chunks:
        logger.warning("generate_answer called with no context chunks for query: %s", query)

    prompt = _build_prompt(query, context_chunks)

    logger.info(
        "Generating answer | query_len=%d | context_chunks=%d",
        len(query), len(context_chunks),
    )

    try:
        response = _client.models.generate_content(
            model=settings.gemini_llm_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=settings.gemini_llm_temperature,
                max_output_tokens=settings.gemini_llm_max_tokens,
            ),
        )
    except Exception:
        logger.error(
            "LLM generation failed | model=%s | query=%s",
            settings.gemini_llm_model, query,
            exc_info=True,
        )
        raise

    logger.info("Answer generated successfully (%d chars)", len(response.text))
    return response.text