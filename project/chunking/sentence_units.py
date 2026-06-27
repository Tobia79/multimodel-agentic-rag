"""Parse markdown text into atomic units that must not be split mid-sentence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

CODE_FENCE_PATTERN = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)
TABLE_ROW_PATTERN = re.compile(r"^\s*\|.+\|\s*$")
LIST_ITEM_PATTERN = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+")
SENTENCE_SPLIT_PATTERN = re.compile(
    r"(?<=[。！？!?])\s+|(?<=[.!?])\s+(?=[A-Z\u4e00-\u9fff])"
)


@dataclass(frozen=True)
class AtomicUnit:
    text: str
    unit_type: str  # prose | list_item | table_row | code_block | reference_line


def join_units(units: List[AtomicUnit], separator: str = "\n") -> str:
    return separator.join(unit.text for unit in units if unit.text)


def parse_reference_line_units(text: str) -> List[AtomicUnit]:
    """Parse reference/bibliography content as one atomic unit per non-empty line."""
    units: List[AtomicUnit] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line:
            units.append(AtomicUnit(text=line, unit_type="reference_line"))
    return units


def parse_atomic_units(text: str) -> List[AtomicUnit]:
    if not text or not text.strip():
        return []

    units: List[AtomicUnit] = []
    pos = 0
    while pos < len(text):
        fence_match = CODE_FENCE_PATTERN.search(text, pos)
        if fence_match and fence_match.start() == pos:
            block = fence_match.group(0).strip()
            if block:
                units.append(AtomicUnit(text=block, unit_type="code_block"))
            pos = fence_match.end()
            continue

        if fence_match and fence_match.start() > pos:
            units.extend(_parse_non_code_block(text[pos : fence_match.start()]))
            pos = fence_match.start()
            continue

        units.extend(_parse_non_code_block(text[pos:]))
        break

    return [unit for unit in units if unit.text.strip()]


def _parse_non_code_block(text: str) -> List[AtomicUnit]:
    units: List[AtomicUnit] = []
    paragraph_lines: List[str] = []

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        paragraph = "\n".join(paragraph_lines).strip()
        paragraph_lines.clear()
        if paragraph:
            units.extend(_split_paragraph_into_units(paragraph))

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            flush_paragraph()
            continue

        if TABLE_ROW_PATTERN.match(line):
            flush_paragraph()
            units.append(AtomicUnit(text=line.strip(), unit_type="table_row"))
            continue

        if LIST_ITEM_PATTERN.match(line):
            flush_paragraph()
            units.append(AtomicUnit(text=line.strip(), unit_type="list_item"))
            continue

        paragraph_lines.append(line)

    flush_paragraph()
    return units


def _split_paragraph_into_units(paragraph: str) -> List[AtomicUnit]:
    sentences = [
        sentence.strip()
        for sentence in SENTENCE_SPLIT_PATTERN.split(paragraph)
        if sentence.strip()
    ]
    if not sentences:
        return [AtomicUnit(text=paragraph.strip(), unit_type="prose")]
    return [AtomicUnit(text=sentence, unit_type="prose") for sentence in sentences]
