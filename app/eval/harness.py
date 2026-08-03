import logging
import time
from typing import Any

from google import genai
from google.genai import types

from app.config import settings
from app.ingestion.embedder import embed_query
from app.retrieval.vector_store import search
from app.retrieval.reranker import rerank
from app.generation.llm_client import generate_answer

logger = logging.getLogger(__name__)

_judge_client = genai.Client(api_key=settings.gemini_api_key)

EVAL_QUESTIONS = [
    {
        "question": "Who is the CEO of Nightingale Robotics?",
        "expected": "Priya Shah",
    },
    {
        "question": "What is the unit price of the SkyCount X3?",
        "expected": "$18,500",
    },
    {
        "question": "What was Nightingale's revenue in fiscal year 2024?",
        "expected": "$47.2 million",
    },
    {
        "question": "What percentage did revenue grow from 2023 to 2024?",
        "expected": "34 percent",
    },
    {
        "question": "What is the total headcount across all three Nightingale offices?",
        "expected": "340 (180 Austin + 95 Berlin + 65 Singapore)",
    },
    {
        "question": "How many SkyCount X3 drones did Bluepeak Logistics deploy, and what reduction in inventory discrepancies did they see?",
        "expected": "45 drones, 22 percent reduction",
    },
    {
        "question": "What is the minimum ceiling height required for SkyCount drones to operate?",
        "expected": "4 meters",
    },
    {
        "question": "How much was the Series C funding round and who led it?",
        "expected": "$60 million, led by Horizon Ventures",
    },
    {
        "question": "What university is Nightingale partnering with for swarm-coordination research, and what is the grant amount?",
        "expected": "University of Texas at Austin, $2.4 million",
    },
    {
        "question": "What is Nightingale's stock ticker symbol?",
        "expected": "NOT_IN_DOCUMENT",
    },
]

JUDGE_PROMPT_TEMPLATE = """You are evaluating a RAG system's answer for factual correctness.

Question: {question}
Expected answer (ground truth): {expected}
System's actual answer: {actual}

Special case: if the expected answer is "NOT_IN_DOCUMENT", the system PASSES only if it
explicitly states it cannot find the answer in the provided documents, and FAILS if it
guesses or fabricates an answer.

Otherwise, judge PASS if the system's answer contains the correct fact(s), even if worded
differently. Judge FAIL if the fact is wrong, missing, or fabricated.

Respond in exactly this format:
VERDICT: PASS or FAIL
REASON: <one sentence>
"""


def _judge(question: str, expected: str, actual: str) -> dict[str, Any]:
    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, expected=expected, actual=actual)

    response = _judge_client.models.generate_content(
        model=settings.gemini_llm_model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )

    text = response.text.strip()
    verdict = "PASS" if "VERDICT: PASS" in text else "FAIL"
    reason = text.split("REASON:")[-1].strip() if "REASON:" in text else text

    return {"verdict": verdict, "reason": reason}


def _check_retrieval(expected: str, chunks: list[dict]) -> bool:
    if expected == "NOT_IN_DOCUMENT":
        return True  # nothing should be retrievable for this
    combined = " ".join(c["content"] for c in chunks).lower()
    # crude but useful signal: does any key token from expected show up in retrieved text
    key_tokens = [t.strip("$,%()") for t in expected.split() if len(t) > 2]
    return any(tok.lower() in combined for tok in key_tokens)


def run_eval() -> None:
    results = []
    passed = 0

    for item in EVAL_QUESTIONS:
        question, expected = item["question"], item["expected"]
        logger.info("Evaluating: %s", question)

        try:
            query_embedding = embed_query(question)
            chunks = search(query_embedding)
            chunks = rerank(question, chunks)
            answer = generate_answer(question, chunks)

            retrieval_ok = _check_retrieval(expected, chunks)
            judged = _judge(question, expected, answer)

            if judged["verdict"] == "PASS":
                passed += 1

            results.append({
                "question": question,
                "expected": expected,
                "actual": answer,
                "verdict": judged["verdict"],
                "reason": judged["reason"],
                "retrieval_found_fact": retrieval_ok,
            })

        except Exception:
            logger.error("Eval question failed to run: %s", question, exc_info=True)
            results.append({
                "question": question,
                "expected": expected,
                "actual": None,
                "verdict": "ERROR",
                "reason": "Exception during pipeline execution",
                "retrieval_found_fact": False,
            })

        time.sleep(1)  # light pacing to avoid rate limits across judge + generation calls

    print("\n" + "=" * 60)
    print(f"EVAL RESULTS: {passed}/{len(EVAL_QUESTIONS)} passed")
    print("=" * 60)
    for r in results:
        status = "✓" if r["verdict"] == "PASS" else "✗"
        retrieval_flag = "" if r["retrieval_found_fact"] else "  [retrieval miss]"
        print(f"{status} [{r['verdict']}] {r['question']}{retrieval_flag}")
        print(f"    expected: {r['expected']}")
        print(f"    got: {r['actual']}")
        print(f"    reason: {r['reason']}\n")


if __name__ == "__main__":
    from app.logging_config import setup_logging
    setup_logging()
    run_eval()