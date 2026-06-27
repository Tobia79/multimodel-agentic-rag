"""Layered retrieval confidence: rerank heuristics + optional LLM (CRAG-style)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence

import config

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "confidence.txt"

_CONFIDENCE_JSON_HINT = (
    '\n\nRespond with a single JSON object only: {"confidence": <0-10 int>, "reasoning": "<string>"}'
)


@dataclass
class RetrievalOutcome:
    documents: List[Any] = field(default_factory=list)
    rerank_scores: List[float] = field(default_factory=list)
    confidence_score: float = 0.0
    confidence_source: str = "none"
    tier: str = "low"
    reasoning: str = ""
    secondary_retrieval_used: bool = False
    from_search: bool = False


def tier_from_score(score: float) -> str:
    if score >= config.CONFIDENCE_HIGH_THRESHOLD:
        return "high"
    if score >= config.CONFIDENCE_LOW_THRESHOLD:
        return "medium"
    return "low"


def preliminary_from_top(top: float) -> float:
    """Map top rerank score to preliminary (0-10) via continuous linear mapping."""
    score_min = config.CONFIDENCE_RERANK_SCORE_MIN
    score_max = config.CONFIDENCE_RERANK_SCORE_MAX
    if score_max <= score_min:
        return 5.0
    normalized = (float(top) - score_min) / (score_max - score_min)
    return max(0.0, min(10.0, normalized * 10.0))


def needs_secondary_retrieval(scores: Sequence[float], result_count: int) -> bool:
    """Whether to expand the retrieval pool and run a second pass."""
    if result_count == 0 or not scores:
        return True
    return float(scores[0]) < config.CONFIDENCE_RERANK_LOW_THRESHOLD


def preliminary_from_rerank(scores: Sequence[float], result_count: int) -> float:
    """Map top rerank score to preliminary confidence (0-10); empty results -> 0."""
    if result_count == 0 or not scores:
        return 0.0
    return preliminary_from_top(float(scores[0]))


def in_llm_gray_zone(preliminary_score: float) -> bool:
    return (
        config.CONFIDENCE_LLM_ENABLED
        and config.CONFIDENCE_RERANK_GRAY_LOW <= preliminary_score <= config.CONFIDENCE_RERANK_GRAY_HIGH
    )


def _load_confidence_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _format_contexts(documents: Sequence[Any], max_chunks: int, max_chars: int) -> str:
    blocks: List[str] = []
    for index, document in enumerate(documents[:max_chunks], start=1):
        text = getattr(document, "page_content", str(document)).strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        blocks.append(f"[{index}] {text}")
    return "\n\n".join(blocks) if blocks else "(empty)"


def evaluate_confidence_llm(
    query: str,
    documents: Sequence[Any],
    llm: Any,
) -> tuple[float, str]:
    """LLM-as-judge retrieval confidence on 0-10 scale."""
    if llm is None:
        return 0.0, "LLM evaluator unavailable"

    prompt = _load_confidence_prompt()
    user_content = (
        f"Query:\n{query}\n\n"
        f"Retrieved excerpts:\n{_format_contexts(documents, config.CONFIDENCE_LLM_MAX_CONTEXT_CHUNKS, 900)}"
    )

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        configured = llm.with_config(temperature=0)
        if config.LLM_PROVIDER == "deepseek":
            response = configured.bind(response_format={"type": "json_object"}).invoke(
                [
                    SystemMessage(content=prompt + _CONFIDENCE_JSON_HINT),
                    HumanMessage(content=user_content),
                ]
            )
            payload = json.loads(response.content)
        else:
            from rag_agent.schemas import RetrievalConfidenceAssessment

            payload = configured.with_structured_output(RetrievalConfidenceAssessment).invoke(
                [SystemMessage(content=prompt), HumanMessage(content=user_content)]
            )
            payload = payload.model_dump()

        score = float(payload.get("confidence", 0))
        score = max(0.0, min(10.0, score))
        reasoning = str(payload.get("reasoning", "")).strip()
        return score, reasoning
    except Exception as exc:
        logger.warning("LLM confidence evaluation failed: %s", exc)
        return 0.0, f"LLM evaluation failed: {exc}"


def finalize_confidence(
    query: str,
    documents: List[Any],
    rerank_scores: List[float],
    preliminary_score: float,
    llm: Optional[Any] = None,
) -> RetrievalOutcome:
    confidence_score = preliminary_score
    confidence_source = "rerank"
    reasoning = ""

    if in_llm_gray_zone(preliminary_score):
        llm_score, llm_reason = evaluate_confidence_llm(query, documents, llm)
        if llm_score > 0 or llm_reason:
            confidence_score = llm_score
            confidence_source = "llm"
            reasoning = llm_reason

    tier = tier_from_score(confidence_score)
    _attach_scores_to_documents(documents, rerank_scores, confidence_score, tier)

    return RetrievalOutcome(
        documents=documents,
        rerank_scores=list(rerank_scores),
        confidence_score=confidence_score,
        confidence_source=confidence_source,
        tier=tier,
        reasoning=reasoning,
        from_search=True,
    )


def _attach_scores_to_documents(
    documents: List[Any],
    rerank_scores: List[float],
    confidence_score: float,
    tier: str,
) -> None:
    for index, document in enumerate(documents):
        metadata = getattr(document, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        if index < len(rerank_scores):
            metadata["rerank_score"] = rerank_scores[index]
        metadata["retrieval_confidence"] = confidence_score
        metadata["retrieval_confidence_tier"] = tier


def merge_retrieval_documents(
    primary: List[Any],
    primary_scores: List[float],
    secondary: List[Any],
    secondary_scores: List[float],
) -> tuple[List[Any], List[float]]:
    """Merge two ranked lists, dedupe by Qdrant point id, keep best rerank score."""
    from core.hybrid_search import document_chunk_id

    best: dict[str, tuple[Any, float]] = {}

    def _ingest(documents, scores):
        for doc, score in zip(documents, scores):
            key = document_chunk_id(doc)
            existing = best.get(key)
            if existing is None or score > existing[1]:
                best[key] = (doc, float(score))

    _ingest(primary, primary_scores)
    _ingest(secondary, secondary_scores)

    merged = sorted(best.values(), key=lambda item: item[1], reverse=True)
    if not merged:
        return [], []
    documents = [item[0] for item in merged]
    scores = [item[1] for item in merged]
    return documents, scores
