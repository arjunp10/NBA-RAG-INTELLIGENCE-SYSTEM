"""LangGraph self-corrective RAG pipeline.

Iterates through RETRIEVAL_CONFIGS in order. Each attempt retrieves chunks,
generates an answer via Gemini, and scores it with RAGAS. If both scores
pass the threshold the result is returned immediately. If all configs are
exhausted without passing, the best-scoring result is returned with
low_confidence=True.

All results are logged to SQLite regardless of outcome.
"""

import logging
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from eval.logger import log_result
from eval.scorer import score_answer
from nba_types import (
    MAX_RETRIES,
    RETRIEVAL_CONFIGS,
    EvalScore,
    QueryResult,
    RetrievalConfig,
    RetrievedChunk,
)
from pipeline.chain import build_chain
from pipeline.retriever import get_retriever

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    question: str
    config_index: int           # which RETRIEVAL_CONFIGS entry to try next
    retry_count: int
    answer: str
    contexts: list[str]         # raw chunk text, passed to RAGAS
    chunks: list[RetrievedChunk]
    score: Optional[EvalScore]
    config_used: Optional[RetrievalConfig]
    best_result: Optional[QueryResult]  # best result seen so far (highest combined score)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def generate_node(state: PipelineState) -> PipelineState:
    """Retrieve chunks and generate an answer for the current config."""
    idx = state["config_index"]
    config = RETRIEVAL_CONFIGS[idx]

    logger.info(
        "Attempt %d — config '%s' (k=%d, chunk_size=%d)",
        state["retry_count"] + 1, config.name, config.k, config.chunk_size,
    )

    retriever = get_retriever(config)
    chain = build_chain(retriever)

    chain_result = chain.invoke({"query": state["question"]})
    answer: str = chain_result["result"]
    source_docs = chain_result.get("source_documents", [])

    chunks = [
        RetrievedChunk(
            content=doc.page_content,
            source=doc.metadata.get("source", ""),
            source_type=doc.metadata.get("source_type", ""),
            score=0.0,  # ChromaDB score not exposed through RetrievalQA
        )
        for doc in source_docs
    ]
    contexts = [doc.page_content for doc in source_docs]

    return {
        **state,
        "answer": answer,
        "contexts": contexts,
        "chunks": chunks,
        "config_used": config,
    }


def score_node(state: PipelineState) -> PipelineState:
    """Score the current answer with RAGAS and update best_result if improved."""
    eval_score = score_answer(state["question"], state["answer"], state["contexts"])

    result = QueryResult(
        question=state["question"],
        answer=state["answer"],
        chunks=state["chunks"],
        score=eval_score,
        config_used=state["config_used"],
        retry_count=state["retry_count"],
        low_confidence=False,  # determined at decision time
    )

    # Keep track of the best result in case we exhaust all configs
    best = state["best_result"]
    if best is None:
        best = result
    else:
        current_combined = eval_score.faithfulness + eval_score.answer_relevance
        best_combined = best.score.faithfulness + best.score.answer_relevance
        if current_combined > best_combined:
            best = result

    return {**state, "score": eval_score, "best_result": best}


def decide_node(state: PipelineState) -> str:
    """Route to END if score passes or configs are exhausted, else retry."""
    if state["score"].passed:
        logger.info("Score passed — returning result after %d attempt(s)", state["retry_count"] + 1)
        return "return_result"

    next_index = state["config_index"] + 1
    if next_index >= len(RETRIEVAL_CONFIGS) or state["retry_count"] >= MAX_RETRIES - 1:
        logger.warning(
            "Max retries reached without passing threshold — returning best result"
        )
        return "return_result"

    logger.warning(
        "Score below threshold (faith=%.3f, rel=%.3f) — retrying with config '%s'",
        state["score"].faithfulness,
        state["score"].answer_relevance,
        RETRIEVAL_CONFIGS[next_index].name,
    )
    return "retry"


def finalize_node(state: PipelineState) -> PipelineState:
    """Mark low_confidence on the best result and log it to SQLite."""
    best = state["best_result"]
    low_confidence = not best.score.passed

    final = QueryResult(
        question=best.question,
        answer=best.answer,
        chunks=best.chunks,
        score=best.score,
        config_used=best.config_used,
        retry_count=state["retry_count"],
        low_confidence=low_confidence,
        timestamp=best.timestamp,
    )

    log_result(final)
    return {**state, "best_result": final}


def increment_retry_node(state: PipelineState) -> PipelineState:
    """Advance to the next config and increment the retry counter."""
    return {
        **state,
        "config_index": state["config_index"] + 1,
        "retry_count": state["retry_count"] + 1,
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("generate", generate_node)
    graph.add_node("score", score_node)
    graph.add_node("increment_retry", increment_retry_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "score")
    graph.add_conditional_edges(
        "score",
        decide_node,
        {
            "return_result": "finalize",
            "retry": "increment_retry",
        },
    )
    graph.add_edge("increment_retry", "generate")
    graph.add_edge("finalize", END)

    return graph


_app = _build_graph().compile()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_query(question: str) -> QueryResult:
    """Run the self-corrective RAG pipeline for a user question.

    Args:
        question: Natural language question about NBA games or players.

    Returns:
        QueryResult with the best answer found, RAGAS scores, and metadata.
    """
    initial_state: PipelineState = {
        "question": question,
        "config_index": 0,
        "retry_count": 0,
        "answer": "",
        "contexts": [],
        "chunks": [],
        "score": None,
        "config_used": None,
        "best_result": None,
    }

    final_state = _app.invoke(initial_state)
    return final_state["best_result"]
