"""Lightweight retrieval metrics (hit_rate, MRR) for RAG evaluation.

Ported from MODULAR-RAG-MCP-SERVER ``CustomEvaluator``; uses parent chunk IDs
as ground-truth anchors for the agentic-rag chunking scheme.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence


CUSTOM_METRIC_NAMES = ("hit_rate", "mrr")

_ID_FIELDS = ("id", "chunk_id", "document_id", "doc_id", "parent_id")


def compute_hit_rate(retrieved_ids: Sequence[str], ground_truth_ids: Sequence[str]) -> float:
    """Return 1.0 if any ground-truth id appears in retrieved ids, else 0.0."""
    if not ground_truth_ids:
        return 0.0
    truth = {str(item) for item in ground_truth_ids}
    return 1.0 if any(str(item) in truth for item in retrieved_ids) else 0.0


def compute_mrr(retrieved_ids: Sequence[str], ground_truth_ids: Sequence[str]) -> float:
    """Reciprocal rank of the first retrieved id that matches ground truth."""
    if not ground_truth_ids:
        return 0.0
    truth = {str(item) for item in ground_truth_ids}
    for rank, item in enumerate(retrieved_ids, start=1):
        if str(item) in truth:
            return 1.0 / rank
    return 0.0


def evaluate_custom_metrics(
    retrieved_ids: Sequence[str],
    ground_truth_ids: Sequence[str],
) -> Dict[str, float]:
    """Compute hit_rate and mrr for a single query."""
    return {
        "hit_rate": compute_hit_rate(retrieved_ids, ground_truth_ids),
        "mrr": compute_mrr(retrieved_ids, ground_truth_ids),
    }


def extract_ids_from_documents(documents: Iterable[Any]) -> List[str]:
    """Extract parent/chunk ids from LangChain Document objects or dicts."""
    ids: List[str] = []
    for doc in documents:
        if doc is None:
            continue
        if isinstance(doc, str):
            if doc.strip():
                ids.append(doc.strip())
            continue
        if isinstance(doc, dict):
            for field in _ID_FIELDS:
                if field in doc and doc[field]:
                    ids.append(str(doc[field]))
                    break
            continue
        metadata = getattr(doc, "metadata", None)
        if isinstance(metadata, dict):
            for field in _ID_FIELDS:
                if field in metadata and metadata[field]:
                    ids.append(str(metadata[field]))
                    break
    return ids
