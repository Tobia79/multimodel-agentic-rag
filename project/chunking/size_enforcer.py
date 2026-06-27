"""Enforce child chunk size limits without splitting sentences when avoidable."""

from __future__ import annotations

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from chunking.sentence_units import AtomicUnit, join_units

_LINE_BASED_TYPES = frozenset({"code_block", "table_row", "list_item", "reference_line"})
_FALLBACK_SEPARATORS = ["\n\n", "\n", "；", ";", "。", ".", "，", ",", " ", ""]


def expand_oversized_units(units: List[AtomicUnit], max_size: int) -> List[AtomicUnit]:
    """Split atomic units longer than max_size for parent chunking."""
    expanded: List[AtomicUnit] = []
    for unit in units:
        if len(unit.text) <= max_size:
            expanded.append(unit)
            continue

        if "\n" in unit.text:
            expanded.extend(_expand_by_lines(unit, max_size))
        else:
            expanded.extend(_expand_long_line(unit, max_size))
    return expanded


def _expand_by_lines(unit: AtomicUnit, max_size: int) -> List[AtomicUnit]:
    expanded: List[AtomicUnit] = []
    lines = [line.strip() for line in unit.text.splitlines() if line.strip()]
    if not lines:
        return _expand_long_line(unit, max_size)

    current_lines: List[str] = []
    current_len = 0
    for line in lines:
        if len(line) > max_size:
            if current_lines:
                expanded.append(
                    AtomicUnit(text="\n".join(current_lines), unit_type=unit.unit_type)
                )
                current_lines = []
                current_len = 0
            expanded.extend(_expand_long_line(AtomicUnit(line, unit.unit_type), max_size))
            continue

        line_len = len(line)
        separator = 1 if current_lines else 0
        projected = current_len + separator + line_len
        if current_lines and projected > max_size:
            expanded.append(
                AtomicUnit(text="\n".join(current_lines), unit_type=unit.unit_type)
            )
            current_lines = [line]
            current_len = line_len
        else:
            current_lines.append(line)
            current_len = projected

    if current_lines:
        expanded.append(
            AtomicUnit(text="\n".join(current_lines), unit_type=unit.unit_type)
        )
    return expanded


def _expand_long_line(unit: AtomicUnit, max_size: int) -> List[AtomicUnit]:
    text = unit.text.strip()
    if len(text) <= max_size:
        return [AtomicUnit(text=text, unit_type=unit.unit_type)]

    chunks: List[AtomicUnit] = []
    start = 0
    while start < len(text):
        end = min(start + max_size, len(text))
        if end < len(text):
            split_at = text.rfind(" ", start, end)
            if split_at <= start:
                split_at = end
            else:
                end = split_at
        piece = text[start:end].strip()
        if piece:
            chunks.append(AtomicUnit(text=piece, unit_type=unit.unit_type))
        start = end if end > start else start + 1
    return chunks


def enforce_child_chunk_sizes(
    units: List[AtomicUnit],
    max_size: int,
    *,
    line_based_only: bool = False,
) -> List[str]:
    if not units:
        return []

    body_chunks = _pack_units(units, max_size, line_based_only=line_based_only)
    return [chunk for chunk in body_chunks if chunk.strip()]


def _pack_units(
    units: List[AtomicUnit],
    max_size: int,
    line_based_only: bool = False,
) -> List[str]:
    chunks: List[str] = []
    current_units: List[AtomicUnit] = []
    current_len = 0

    for unit in units:
        unit_len = len(unit.text)
        if unit_len > max_size:
            if current_units:
                chunks.append(join_units(current_units))
                current_units = []
                current_len = 0
            chunks.extend(_split_oversized_unit(unit, max_size))
            continue

        separator_len = 1 if current_units else 0
        projected = current_len + separator_len + unit_len
        if current_units and projected > max_size:
            chunks.append(join_units(current_units))
            current_units = [unit]
            current_len = unit_len
        else:
            current_units.append(unit)
            current_len = projected

    if current_units:
        chunks.append(join_units(current_units))

    normalized: List[str] = []
    for chunk in chunks:
        if len(chunk) <= max_size:
            normalized.append(chunk)
        elif line_based_only:
            normalized.extend(_merge_lines_to_chunks(chunk, max_size))
        else:
            normalized.extend(_force_split_text(chunk, max_size))
    return normalized


def _split_oversized_unit(unit: AtomicUnit, max_size: int) -> List[str]:
    if unit.unit_type in _LINE_BASED_TYPES or "\n" in unit.text:
        return _merge_lines_to_chunks(unit.text, max_size)
    return _force_split_text(unit.text, max_size)


def _merge_lines_to_chunks(text: str, max_size: int) -> List[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return _force_split_text(text, max_size)

    chunks: List[str] = []
    current_lines: List[str] = []
    current_len = 0

    for line in lines:
        line_len = len(line)
        separator = 1 if current_lines else 0
        projected = current_len + separator + line_len

        if current_lines and projected > max_size:
            chunks.append("\n".join(current_lines))
            current_lines = [line]
            current_len = line_len
        else:
            current_lines.append(line)
            current_len = projected

        if len(current_lines) == 1 and current_len > max_size:
            chunks.extend(_force_split_text(current_lines[0], max_size))
            current_lines = []
            current_len = 0

    if current_lines:
        remaining = "\n".join(current_lines)
        if len(remaining) <= max_size:
            chunks.append(remaining)
        elif len(current_lines) > 1:
            chunks.extend(_merge_lines_to_chunks(remaining, max_size))
        else:
            chunks.extend(_force_split_text(remaining, max_size))

    return chunks


def _force_split_text(text: str, max_size: int) -> List[str]:
    if len(text) <= max_size:
        return [text]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_size,
        chunk_overlap=0,
        separators=_FALLBACK_SEPARATORS,
    )
    parts = splitter.split_text(text)
    return [part for part in parts if part.strip()]
