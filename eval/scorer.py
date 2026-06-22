"""Scores a RAG answer using RAGAS faithfulness and answer_relevancy metrics.

Configured to use Gemini 1.5 Flash as the judge LLM and Google text-embedding-004
for embeddings, avoiding any OpenAI dependency.
"""

import logging
import os
from typing import List

from datasets import Dataset
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, faithfulness

from nba_types import EMBEDDING_MODEL, GEMINI_MODEL, SCORE_THRESHOLD, EvalScore

load_dotenv()

logger = logging.getLogger(__name__)


def _get_ragas_llm() -> LangchainLLMWrapper:
    """Wrap Gemini in a RAGAS-compatible LLM wrapper."""
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.0,
    )
    return LangchainLLMWrapper(llm)


def _get_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    """Wrap Google embeddings in a RAGAS-compatible embeddings wrapper."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )
    return LangchainEmbeddingsWrapper(embeddings)


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

        faithfulness.llm = ragas_llm
        answer_relevancy.llm = ragas_llm
        answer_relevancy.embeddings = ragas_embeddings

        dataset = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        })

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
        )

        faith_score = float(result["faithfulness"])
        relevance_score = float(result["answer_relevancy"])
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
