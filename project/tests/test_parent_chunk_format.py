"""Tests for parent chunk presentation formatting."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_agent.chunk_format import build_section_hierarchy, format_parent_chunk_for_agent


def test_build_section_hierarchy_merged_headers():
    meta = {"H2": "Intro -> Details -> Summary"}
    assert build_section_hierarchy(meta) == "Intro > Details > Summary"


def test_build_section_hierarchy_h1_h2_h3():
    meta = {"H1": "Chapter 1", "H2": "Section A", "H3": "Subsection"}
    assert build_section_hierarchy(meta) == "Chapter 1 > Section A > Subsection"


def test_format_parent_chunk_for_agent_structure():
    text = format_parent_chunk_for_agent(
        "doc_parent_0",
        "Body paragraph here.",
        {
            "parent_id": "doc_parent_0",
            "source": "doc.pdf",
            "H1": "Chapter",
            "H2": "Section",
            "title": "Section Title",
            "summary": "Short summary.",
            "tags": ["alpha", "beta"],
        },
    )
    assert "Parent ID: doc_parent_0" in text
    assert "File Name: doc.pdf" in text
    assert "Section: Chapter > Section" in text
    assert "Title: Section Title" in text
    assert "Summary: Short summary." in text
    assert "Tags: alpha, beta" in text
    assert "---\nContent:\nBody paragraph here." in text


def test_format_parent_chunk_omits_empty_metadata_lines():
    text = format_parent_chunk_for_agent(
        "doc_parent_1",
        "Only content.",
        {"source": "doc.pdf"},
    )
    assert "Section:" not in text
    assert "Title:" not in text
    assert "Summary:" not in text
    assert "Tags:" not in text
    assert text.endswith("Content:\nOnly content.")


if __name__ == "__main__":
    test_build_section_hierarchy_merged_headers()
    test_build_section_hierarchy_h1_h2_h3()
    test_format_parent_chunk_for_agent_structure()
    test_format_parent_chunk_omits_empty_metadata_lines()
    print("All parent chunk format tests passed.")
