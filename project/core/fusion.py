"""Reciprocal Rank Fusion (RRF) for combining dense and sparse rankings."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

import config

logger = logging.getLogger(__name__)


@dataclass
class RankedDocument:
    chunk_id: str
    score: float
    document: Any


class RRFFusion:
    """Fuse multiple ranked lists with RRF: score(d) = sum 1 / (k + rank(d))."""

    def __init__(self, k: Optional[int] = None) -> None:
        self.k = config.RRF_K if k is None else k
        if not isinstance(self.k, int) or self.k <= 0:
            raise ValueError(f"RRF k must be a positive integer, got {self.k}")

    def fuse(
        self,
        ranking_lists: List[List[RankedDocument]],
        top_k: Optional[int] = None,
    ) -> List[RankedDocument]:
        non_empty = [ranking for ranking in ranking_lists if ranking]
        if not non_empty:
            return []
        if len(non_empty) == 1:
            return non_empty[0][:top_k] if top_k else non_empty[0]

        rrf_scores: dict[str, float] = {}
        chunk_data: dict[str, RankedDocument] = {}

        for ranking in non_empty:
            for rank, item in enumerate(ranking, start=1):
                if item.chunk_id not in rrf_scores:
                    rrf_scores[item.chunk_id] = 0.0
                    chunk_data[item.chunk_id] = item
                rrf_scores[item.chunk_id] += 1.0 / (self.k + rank)

        fused = [
            RankedDocument(
                chunk_id=chunk_id,
                score=rrf_score,
                document=chunk_data[chunk_id].document,
            )
            for chunk_id, rrf_score in rrf_scores.items()
        ]
        fused.sort(key=lambda item: (-item.score, item.chunk_id))

        if top_k is not None and top_k > 0:
            fused = fused[:top_k]

        logger.debug("RRF fused %s lists into %s results (top_k=%s)", len(non_empty), len(fused), top_k)
        return fused
