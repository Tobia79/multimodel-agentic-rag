"""Hybrid retrieval: dense + sparse + RRF fusion + rerank + layered confidence."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import config
from core.confidence import (
    RetrievalOutcome,
    finalize_confidence,
    merge_retrieval_documents,
    needs_secondary_retrieval,
    preliminary_from_rerank,
)
from core.hybrid_search import HybridSearchEngine
from core.reranker import get_reranker

logger = logging.getLogger(__name__)


def _candidate_id(document: Any, index: int) -> str:
    metadata = getattr(document, "metadata", None) or {}
    parent_id = metadata.get("parent_id")
    if parent_id:
        return f"{parent_id}::{index}"
    return f"chunk_{index}"


def _documents_to_candidates(documents: List[Any]) -> List[dict]:
    candidates = []
    for index, document in enumerate(documents):
        candidates.append(
            {
                "id": _candidate_id(document, index),
                "text": getattr(document, "page_content", str(document)),
                "metadata": dict(getattr(document, "metadata", {}) or {}),
                "document": document,
            }
        )
    return candidates


def _candidates_to_documents(candidates: List[dict]) -> List[Any]:
    return [candidate["document"] for candidate in candidates if candidate.get("document") is not None]


def _rerank_enabled() -> bool:
    return config.RERANK_ENABLED and config.RERANK_PROVIDER not in {"none", "disabled"}


def fusion_output_k(
    final_k: int,
    *,
    multiplier: int = 1,
    dense_top_k: Optional[int] = None,
    sparse_top_k: Optional[int] = None,
) -> int:
    """RRF pool size before rerank; multiplier expands candidate pool for secondary retrieval."""
    dense_k = dense_top_k if dense_top_k is not None else config.DENSE_TOP_K
    sparse_k = sparse_top_k if sparse_top_k is not None else config.SPARSE_TOP_K
    if final_k < 1:
        return 0
    scaled = final_k * max(1, multiplier)
    if _rerank_enabled():
        pool = max(scaled, scaled * config.RERANK_CANDIDATE_MULTIPLIER)
        return max(pool, dense_k, sparse_k)
    return max(scaled, dense_k, sparse_k)


def _rerank_documents(
    query: str,
    fused_docs: List[Any],
    final_k: int,
) -> Tuple[List[Any], List[float]]:
    if not fused_docs:
        return [], []

    if not _rerank_enabled() or len(fused_docs) == 1:
        docs = fused_docs[:final_k]
        return docs, [0.0] * len(docs)

    candidates = _documents_to_candidates(fused_docs)
    try:
        reranked = get_reranker().rerank(query, candidates, top_k=final_k)
        scores = [float(item.get("rerank_score", 0.0)) for item in reranked]
        return _candidates_to_documents(reranked), scores
    except Exception as exc:
        logger.warning("Rerank failed, using RRF order: %s", exc)
        docs = fused_docs[:final_k]
        return docs, [0.0] * len(docs)


def _single_retrieval_pass(
    collection: Any,
    query: str,
    final_k: int,
    *,
    pool_multiplier: int = 1,
    dense_top_k: Optional[int] = None,
    sparse_top_k: Optional[int] = None,
) -> Tuple[List[Any], List[float]]:
    hybrid = HybridSearchEngine(collection)
    fused_docs = hybrid.search(
        query,
        fusion_top_k=fusion_output_k(
            final_k,
            multiplier=pool_multiplier,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
        ),
        dense_top_k=dense_top_k,
        sparse_top_k=sparse_top_k,
    )
    return _rerank_documents(query, fused_docs, final_k)


def retrieve_top_k(
    collection: Any,
    query: str,
    limit: Optional[int] = None,
) -> List[Any]:
    """Pure retrieval: hybrid + RRF + rerank → Top-K (no confidence routing)."""
    final_k = config.FUSION_TOP_K if limit is None else limit
    if final_k < 1 or not query or not query.strip():
        return []
    documents, _scores = _single_retrieval_pass(collection, query, final_k)
    return documents


def retrieve_with_confidence(
    collection: Any,
    query: str,
    limit: Optional[int] = None,
    *,
    llm: Any = None,
    enable_confidence: bool = True,
) -> RetrievalOutcome:
    """Dense + sparse + RRF + rerank, then optional layered confidence."""
    final_k = config.FUSION_TOP_K if limit is None else limit
    if final_k < 1 or not query or not query.strip():
        return RetrievalOutcome(from_search=True)

    documents, scores = _single_retrieval_pass(collection, query, final_k)

    if not enable_confidence or not config.CONFIDENCE_ENABLED:
        return RetrievalOutcome(
            documents=documents,
            rerank_scores=scores,
            confidence_score=7.0 if documents else 0.0,
            confidence_source="disabled",
            tier="high" if documents else "low",
            from_search=True,
        )

    secondary_used = False
    if needs_secondary_retrieval(scores, len(documents)) and config.CONFIDENCE_SECONDARY_RETRIEVAL:
        expanded_docs, expanded_scores = _single_retrieval_pass(
            collection,
            query,
            final_k,
            pool_multiplier=config.CONFIDENCE_SECONDARY_MULTIPLIER,
            dense_top_k=config.CONFIDENCE_SECONDARY_DENSE_TOP_K,
            sparse_top_k=config.CONFIDENCE_SECONDARY_SPARSE_TOP_K,
        )
        documents, scores = merge_retrieval_documents(
            documents,
            scores,
            expanded_docs,
            expanded_scores,
        )
        documents = documents[:final_k]
        scores = scores[:final_k]
        secondary_used = True

    preliminary = preliminary_from_rerank(scores, len(documents))
    outcome = finalize_confidence(query, documents, scores, preliminary, llm=llm)
    outcome.secondary_retrieval_used = secondary_used
    return outcome


def retrieve_child_documents(
    collection: Any,
    query: str,
    limit: Optional[int] = None,
    *,
    llm: Any = None,
    enable_confidence: bool = True,
) -> List[Any]:
    """Retrieve child chunks; set enable_confidence=False for evaluation-style Top-K only."""
    if not enable_confidence:
        return retrieve_top_k(collection, query, limit)
    return retrieve_with_confidence(
        collection,
        query,
        limit,
        llm=llm,
        enable_confidence=True,
    ).documents
