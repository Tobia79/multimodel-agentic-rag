"""Sequential child-chunk overlap: last sentence/clause from the previous child."""

from __future__ import annotations

from typing import Callable, List

_SENTENCE_MARKERS = ".!?"
_SEPARATOR = "\n"


def _marker_positions(text: str, markers: str) -> List[int]:
    return [index for index, char in enumerate(text) if char in markers]


def extract_tail_by_sentence_en(text: str) -> str:
    """Take the last complete English sentence (. ! ?) from the previous chunk."""
    stripped = text.strip()
    if not stripped:
        return ""

    positions = _marker_positions(stripped, _SENTENCE_MARKERS)
    if not positions:
        return ""

    if len(positions) == 1:
        return stripped

    start = positions[-2] + 1
    return stripped[start:].strip()


def extract_tail_by_clause_en(text: str) -> str:
    """Take the trailing segment after the last English comma."""
    stripped = text.strip()
    if not stripped:
        return ""

    comma_index = stripped.rfind(",")
    if comma_index == -1 or comma_index >= len(stripped) - 1:
        return ""

    return stripped[comma_index + 1 :].strip()


def prepend_child_overlap(previous_chunk: str, current_chunk: str, max_size: int) -> str:
    """
    Prepend overlap from the previous child chunk.

    Prefer the last English sentence (. ! ?); if the result exceeds max_size,
    degrade to the last comma-delimited clause.
    """
    body = current_chunk.strip()
    if not body or not previous_chunk.strip():
        return body

    extractors: List[Callable[[str], str]] = [
        extract_tail_by_sentence_en,
        extract_tail_by_clause_en,
    ]

    for extract in extractors:
        overlap = extract(previous_chunk)
        if not overlap:
            continue
        combined = overlap + _SEPARATOR + body
        if len(combined) <= max_size:
            return combined

    return body


def apply_sequential_child_overlap(chunks: List[str], max_size: int) -> List[str]:
    if not chunks:
        return []

    result = [chunks[0].strip()]
    for chunk in chunks[1:]:
        merged = prepend_child_overlap(result[-1], chunk, max_size)
        result.append(merged)

    return [chunk for chunk in result if chunk]
