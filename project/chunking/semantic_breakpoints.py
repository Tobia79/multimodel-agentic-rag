"""Contextual embedding similarity and percentile-based semantic breakpoints."""

from __future__ import annotations

from typing import List, Set

import config
import numpy as np
from chunking.sentence_units import AtomicUnit

_PROSE_TYPE = "prose"


class SemanticBreakpointDetector:
    _model = None

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            from sentence_transformers import SentenceTransformer

            cls._model = SentenceTransformer(config.DENSE_MODEL)
        return cls._model

    @staticmethod
    def _contextual_texts(units: List[AtomicUnit], buffer_size: int) -> List[str]:
        texts: List[str] = []
        for index in range(len(units)):
            start = max(0, index - buffer_size)
            end = min(len(units), index + buffer_size + 1)
            texts.append("\n".join(unit.text for unit in units[start:end]))
        return texts

    @classmethod
    def adjacent_similarities(
        cls,
        units: List[AtomicUnit],
        buffer_size: int,
    ) -> List[float]:
        if len(units) < 2:
            return []

        model = cls._get_model()
        embeddings = model.encode(
            cls._contextual_texts(units, buffer_size),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        similarities: List[float] = []
        for index in range(len(embeddings) - 1):
            score = float(np.dot(embeddings[index], embeddings[index + 1]))
            similarities.append(score)
        return similarities

    @staticmethod
    def percentile_threshold(similarities: List[float], percentile: float) -> float:
        if not similarities:
            return 1.0
        return float(np.percentile(similarities, percentile))

    @classmethod
    def _detect_prose_semantic_breakpoints(
        cls,
        prose_units: List[AtomicUnit],
        buffer_size: int,
        percentile: float,
    ) -> Set[int]:
        """Semantic breakpoints within a contiguous prose run (local indices)."""
        if len(prose_units) < 2:
            return set()

        similarities = cls.adjacent_similarities(prose_units, buffer_size)
        threshold = cls.percentile_threshold(similarities, percentile)
        return {
            index
            for index, score in enumerate(similarities)
            if score < threshold
        }

    @classmethod
    def detect_breakpoints(
        cls,
        units: List[AtomicUnit],
        buffer_size: int,
        percentile: float,
    ) -> Set[int]:
        """
        Return indices i where a break occurs after unit i (before unit i+1).

        List items, table rows, and code blocks are hard boundaries and never
        participate in similarity scoring. Semantic breakpoints are computed
        only between adjacent prose units.
        """
        if len(units) < 2:
            return set()

        breakpoints: Set[int] = set()

        for index in range(len(units) - 1):
            left = units[index].unit_type
            right = units[index + 1].unit_type
            if left != _PROSE_TYPE or right != _PROSE_TYPE:
                breakpoints.add(index)

        prose_start = 0
        while prose_start < len(units):
            if units[prose_start].unit_type != _PROSE_TYPE:
                prose_start += 1
                continue

            prose_end = prose_start
            while prose_end < len(units) and units[prose_end].unit_type == _PROSE_TYPE:
                prose_end += 1

            prose_run = units[prose_start:prose_end]
            run_breaks = cls._detect_prose_semantic_breakpoints(
                prose_run,
                buffer_size,
                percentile,
            )
            breakpoints.update(prose_start + local_index for local_index in run_breaks)
            prose_start = prose_end

        return breakpoints

    @classmethod
    def split_segments(
        cls,
        units: List[AtomicUnit],
        breakpoints: Set[int],
    ) -> List[List[AtomicUnit]]:
        if not units:
            return []

        segments: List[List[AtomicUnit]] = []
        current: List[AtomicUnit] = []
        for index, unit in enumerate(units):
            current.append(unit)
            if index in breakpoints:
                segments.append(current)
                current = []
        if current:
            segments.append(current)
        return segments

    @classmethod
    def build_parent_texts(
        cls,
        units: List[AtomicUnit],
        breakpoints: Set[int],
        max_size: int,
    ) -> List[str]:
        if not units:
            return []

        parents: List[str] = []
        current_units: List[AtomicUnit] = []

        def flush_current() -> None:
            if not current_units:
                return
            parents.append("\n\n".join(item.text for item in current_units))
            current_units.clear()

        for index, unit in enumerate(units):
            separator_len = 2 if current_units else 0
            projected_len = (
                sum(len(item.text) for item in current_units)
                + separator_len
                + len(unit.text)
            )

            if current_units and projected_len > max_size:
                flush_current()

            current_units.append(unit)

            if index in breakpoints and index < len(units) - 1:
                flush_current()

        flush_current()
        return parents
