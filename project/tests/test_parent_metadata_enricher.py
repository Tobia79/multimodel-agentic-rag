"""Tests for parent metadata LLM parsing and enrichment."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document

from ingestion.transform.metadata_enricher import MetadataEnricher


def test_normalize_tags_caps_at_ten():
    enricher = MetadataEnricher(use_llm=False, parent_use_llm=False)
    tags = [f"tag{i}" for i in range(15)]
    normalized = enricher._normalize_tags(tags, 10)
    assert len(normalized) == 10
    assert normalized[0] == "tag0"
    assert normalized[-1] == "tag9"


def test_parse_parent_llm_response():
    enricher = MetadataEnricher(use_llm=False, parent_use_llm=False)
    response = (
        "Summary: This section explains JavaScript basics and syntax.\n"
        "Tags: JavaScript, syntax, variables, functions, DOM, events, "
        "objects, arrays, loops, types, extra"
    )
    parsed = enricher._parse_parent_llm_response(response)
    assert "JavaScript basics" in parsed["summary"]
    assert len(parsed["tags"]) == 10
    assert "extra" not in parsed["tags"]


def test_enrich_parent_single_uses_llm_summary_and_tags():
    enricher = MetadataEnricher(use_llm=False, parent_use_llm=True, llm=MagicMock())
    enricher._llm = enricher.llm

    with patch.object(
        enricher,
        "_llm_enrich_parent",
        return_value={
            "summary": "LLM generated parent summary.",
            "tags": ["alpha", "beta", "gamma"],
        },
    ):
        parent_doc = Document(
            page_content="## Intro\nSome body text about alpha concepts.",
            metadata={"H2": "Intro", "source": "doc.pdf"},
        )
        (_, enriched), enriched_by = enricher._enrich_parent_single("doc_parent_0", parent_doc)

    assert enriched_by == "llm"
    assert enriched.metadata["summary"] == "LLM generated parent summary."
    assert enriched.metadata["tags"] == ["alpha", "beta", "gamma"]
    assert enriched.metadata["enriched_by"] == "llm"
    assert enriched.metadata["title"]


def test_enrich_parent_single_falls_back_to_rules():
    enricher = MetadataEnricher(use_llm=False, parent_use_llm=True, llm=MagicMock())

    with patch.object(enricher, "_llm_enrich_parent", return_value=None):
        parent_doc = Document(
            page_content="## Section\nFirst sentence. Second sentence.",
            metadata={"H2": "Section"},
        )
        (_, enriched), enriched_by = enricher._enrich_parent_single("doc_parent_1", parent_doc)

    assert enriched_by == "rule"
    assert enriched.metadata["enriched_by"] == "rule"
    assert enriched.metadata["summary"]
    assert isinstance(enriched.metadata["tags"], list)


if __name__ == "__main__":
    test_normalize_tags_caps_at_ten()
    test_parse_parent_llm_response()
    test_enrich_parent_single_uses_llm_summary_and_tags()
    test_enrich_parent_single_falls_back_to_rules()
    print("All parent metadata enricher tests passed.")
