"""Scores a RAG answer using RAGAS faithfulness and answer_relevancy metrics.

Configured to use Gemini 1.5 Flash as the judge LLM and Google text-embedding-004
for embeddings, avoiding any OpenAI dependency.

Uses the RAGAS v0.2+ API: llm/embeddings are passed directly to evaluate()
rather than mutated on global metric singletons.
"""

import logging
import math
import os
from typing import List

from datasets import Dataset
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, Faithfulness

from nba_types import EMBEDDING_MODEL, GEMINI_MODEL, SCORE_THRESHOLD, EvalScore

load_dotenv()

logger = logging.getLogger(__name__)


def _get_ragas_llm() -> LangchainLLMWrapper:
    """Wrap Gemini in a RAGAS-compatible LLM wrapper."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your .env file.")
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=0.0,
    )
    return LangchainLLMWrapper(llm)


def _get_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    """Wrap Google embeddings in a RAGAS-compatible embeddings wrapper."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your .env file.")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )
    return LangchainEmbeddingsWrapper(embeddings)


def _safe_float(value: object) -> float:
    """Extract a scalar float from a RAGAS result value (handles list or scalar)."""
    if isinstance(value, list):
        value = value[0]
    try:
        f = float(value)
        return 0.0 if math.isnan(f) else f
    except (TypeError, ValueError):
        return 0.0


def score_answer(question: str, answer: str, contexts: List[str]) -> EvalScore:
    """Run RAGAS faithfulness and answer_relevancy scoring on a single Q&A pair.

    Args:
        question: The original user question.
        answer: The generated answer from Gemini.
        contexts: List of raw text chunks used to generate the answer.

    Returns:
        EvalScore with faithfulness, answer_relevance, and passed flag.
    """
    try:
        ragas_llm = _get_ragas_llm()
        ragas_embeddings = _get_ragas_embeddings()

        # Instantiate fresh metric objects per call — avoids global state mutation
        # and is compatible with RAGAS v0.2+
        faithfulness_metric = Faithfulness(llm=ragas_llm)
        answer_relevancy_metric = AnswerRelevancy(
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )

        dataset = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        })

        result = evaluate(
            dataset,
            metrics=[faithfulness_metric, answer_relevancy_metric],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )

        faith_score = _safe_float(result["faithfulness"])
        relevance_score = _safe_float(result["answer_relevancy"])
        passed = faith_score >= SCORE_THRESHOLD and relevance_score >= SCORE_THRESHOLD

        logger.info(
            "RAGAS scores — faithfulness: %.3f, answer_relevancy: %.3f, passed: %s",
            faith_score, relevance_score, passed,
        )
        return EvalScore(
            faithfulness=faith_score,
            answer_relevance=relevance_score,
            passed=passed,
        )

    except Exception as exc:
        logger.error("RAGAS scoring failed: %s", exc)
        return EvalScore(faithfulness=0.0, answer_relevance=0.0, passed=False)
