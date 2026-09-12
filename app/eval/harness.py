import logging
import time

from ragas import evaluate, EvaluationDataset
from ragas.run_config import RunConfig
from ragas.llms import BaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_core.outputs import LLMResult, Generation

from google import genai
from google.genai import types

from app.config import settings
from app.ingestion.embedder import embed_query
from app.retrieval.vector_store import search
from app.retrieval.reranker import rerank
from app.generation.llm_client import generate_answer

logger = logging.getLogger(__name__)

_genai_client = genai.Client(api_key=settings.gemini_api_key)

# Gemini free/standard tiers here cap at 15 RPM. Every question makes 2 Gemini calls
# (generate_answer inside the pipeline, then the RAGAS judge call per metric on top),
# so this sleep is between-question pacing on top of that, not a substitute for it.
SECONDS_BETWEEN_QUESTIONS = 8


# --- Wrap the same Gemini client/model your app already uses as the RAGAS judge ---
class GeminiRagasLLM(BaseRagasLLM):
    def generate_text(self, prompt, n=1, temperature=1e-8, stop=None, callbacks=None) -> LLMResult:
        prompt_str = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        generations = []
        for _ in range(n):
            time.sleep(4)  # 15 RPM cap — pace every individual judge call, not just per-question
            response = _genai_client.models.generate_content(
                model=settings.gemini_llm_model,
                contents=prompt_str,
                config=types.GenerateContentConfig(temperature=temperature or 0.0),
            )
            generations.append(Generation(text=response.text))
        return LLMResult(generations=[generations])

    async def agenerate_text(self, prompt, n=1, temperature=1e-8, stop=None, callbacks=None) -> LLMResult:
        return self.generate_text(prompt, n, temperature, stop, callbacks)

    def is_finished(self, response: LLMResult) -> bool:
        # Gemini responses here aren't token-capped mid-generation for our short judge
        # prompts, so treat every response as complete.
        return True


# --- Wrap your existing embed_query so context_precision/recall score against the same
#     embeddings your retrieval actually uses ---
class AppEmbeddings(BaseRagasEmbedding):
    def embed_text(self, text: str, **kwargs):
        return embed_query(text)

    async def aembed_text(self, text: str, **kwargs):
        return self.embed_text(text)


EVAL_QUESTIONS = [
    {"question": "What are Northstar's standard customer support hours?", "ground_truth": "Monday through Friday, 8:00 a.m. to 6:00 p.m. Central Time"},
    {"question": "How much PTO does a U.S. employee on the standard plan get in their first three years?", "ground_truth": "20 days per calendar year"},
    {"question": "What is the standard lodging limit for domestic U.S. business travel?", "ground_truth": "$250 per night before taxes and mandatory fees"},
    {"question": "Is alcohol reimbursable during business travel?", "ground_truth": "No, alcohol is not reimbursable"},
    {"question": "What approval is required for a purchase between $5,001 and $25,000?", "ground_truth": "Manager and Finance approval"},
    {"question": "How long are standard production backups retained?", "ground_truth": "35 days, unless a contractual requirement specifies otherwise"},
    {"question": "Within what time frame should a Severity 2 incident be acknowledged?", "ground_truth": "Within 30 minutes during covered hours"},
    {"question": "What percentage does Northstar match on employee 401(k) contributions?", "ground_truth": "50% of employee contributions on the first 6% of eligible compensation"},
    {"question": "Up to what amount can a support representative approve a service credit without escalation?", "ground_truth": "$250"},
    {"question": "How often do employees generally receive a formal performance review?", "ground_truth": "Twice each year"},
]

ABSTENTION_CASE = {
    "question": "What is Northstar Analytics' stock ticker symbol?",
    "expected_behavior": "must explicitly say it cannot find this in the documents",
}


def _run_pipeline(question: str) -> tuple[str, list[str]]:
    query_embedding = embed_query(question)
    chunks = search(query_embedding)
    chunks = rerank(question, chunks)
    answer = generate_answer(question, chunks)
    contexts = [c["content"] for c in chunks]
    return answer, contexts


def _check_abstention(answer: str) -> bool:
    refusal_signals = ["cannot find", "not available", "no information", "not mentioned", "don't have", "not provided", "unable to find"]
    return any(sig in answer.lower() for sig in refusal_signals)


def build_dataset() -> EvaluationDataset:
    rows = []
    for item in EVAL_QUESTIONS:
        logger.info("Running pipeline: %s", item["question"])
        answer, contexts = _run_pipeline(item["question"])
        rows.append({
            "user_input": item["question"],
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": item["ground_truth"],
        })
        time.sleep(SECONDS_BETWEEN_QUESTIONS)
    return EvaluationDataset.from_list(rows)


def run_eval() -> None:
    dataset = build_dataset()

    # 15 RPM = 1 call per 4s. max_workers=1 forces judge calls fully sequential;
    # max_retries/max_wait give it room to back off instead of hammering on a 429.
    run_config = RunConfig(
        max_workers=1,
        max_retries=5,
        max_wait=60,
    )

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=GeminiRagasLLM(),
        embeddings=AppEmbeddings(),
        run_config=run_config,
    )

    print("\n" + "=" * 60)
    print("RAGAS RESULTS")
    print("=" * 60)
    print(result)
    df = result.to_pandas()
    print(df.to_string())
    df.to_csv("ragas_eval_results.csv", index=False)

    print("\n" + "-" * 60)
    print("ABSTENTION CHECK (not scored by RAGAS)")
    answer, _ = _run_pipeline(ABSTENTION_CASE["question"])
    abstained = _check_abstention(answer)
    status = "PASS" if abstained else "FAIL"
    print(f"[{status}] {ABSTENTION_CASE['question']}")
    print(f"  answer: {answer}")


if __name__ == "__main__":
    from app.logging_config import setup_logging
    setup_logging()
    run_eval()