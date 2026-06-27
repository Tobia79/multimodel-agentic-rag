"""Metadata enrichment: rule-based + optional LLM enhancement."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config
from langchain_core.documents import Document

from ingestion.llm_helpers import create_ingestion_llm, invoke_text_llm
from ingestion.trace import IngestionTrace
from ingestion.transform.base_transform import BaseTransform

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "metadata_enrichment.txt"
PARENT_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "parent_metadata_enrichment.txt"


class MetadataEnricher(BaseTransform):
    def __init__(
        self,
        use_llm: Optional[bool] = None,
        llm=None,
        *,
        parent_use_llm: Optional[bool] = None,
    ):
        self.use_llm = (
            config.INGESTION_METADATA_ENRICHER_USE_LLM if use_llm is None else use_llm
        )
        self.parent_use_llm = (
            config.INGESTION_PARENT_METADATA_USE_LLM
            if parent_use_llm is None
            else parent_use_llm
        )
        self._llm = llm
        self._prompt_template: Optional[str] = None
        self._parent_prompt_template: Optional[str] = None

    @property
    def llm(self):
        needs_llm = self.use_llm or self.parent_use_llm
        if needs_llm and self._llm is None:
            try:
                self._llm = create_ingestion_llm()
            except Exception as exc:
                logger.warning("MetadataEnricher LLM init failed: %s", exc)
                self.use_llm = False
                self.parent_use_llm = False
        return self._llm

    def transform(
        self,
        documents: List[Document],
        trace: Optional[IngestionTrace] = None,
    ) -> List[Document]:
        if not documents:
            return []
        if self.use_llm and self.llm:
            return self._transform_parallel(documents, trace)
        return self._transform_sequential(documents, trace)

    def _transform_single(
        self,
        document: Document,
        trace: Optional[IngestionTrace] = None,
    ) -> Tuple[Document, str]:
        text = document.page_content or ""
        rule_metadata = self._rule_based_enrich(text)
        enriched_by = "rule"
        enriched_metadata = rule_metadata

        if self.use_llm and self.llm:
            llm_metadata = self._llm_enrich(text)
            if llm_metadata:
                enriched_metadata = llm_metadata
                enriched_by = "llm"

        metadata = {
            **(document.metadata or {}),
            **enriched_metadata,
            "enriched_by": enriched_by,
        }
        return Document(page_content=text, metadata=metadata), enriched_by

    def _transform_parallel(
        self,
        documents: List[Document],
        trace: Optional[IngestionTrace] = None,
    ) -> List[Document]:
        max_workers = min(config.INGESTION_LLM_MAX_WORKERS, len(documents))
        results: List[Optional[Document]] = [None] * len(documents)
        llm_count = 0
        rule_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(self._transform_single, doc, trace): idx
                for idx, doc in enumerate(documents)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    enriched_doc, enriched_by = future.result()
                    results[idx] = enriched_doc
                    if enriched_by == "llm":
                        llm_count += 1
                    else:
                        rule_count += 1
                except Exception as exc:
                    logger.error("Metadata enrich failed: %s", exc)
                    results[idx] = documents[idx]

        if trace:
            trace.record_stage(
                "metadata_enricher",
                {
                    "total_chunks": len(documents),
                    "llm_enhanced_count": llm_count,
                    "rule_count": rule_count,
                    "use_llm": self.use_llm,
                },
            )
        return [doc for doc in results if doc is not None]

    def _transform_sequential(
        self,
        documents: List[Document],
        trace: Optional[IngestionTrace] = None,
    ) -> List[Document]:
        enriched_docs = []
        llm_count = 0
        rule_count = 0

        for document in documents:
            try:
                enriched_doc, enriched_by = self._transform_single(document, trace)
                enriched_docs.append(enriched_doc)
                if enriched_by == "llm":
                    llm_count += 1
                else:
                    rule_count += 1
            except Exception as exc:
                logger.error("Metadata enrich failed: %s", exc)
                enriched_docs.append(document)

        if trace:
            trace.record_stage(
                "metadata_enricher",
                {
                    "total_chunks": len(documents),
                    "llm_enhanced_count": llm_count,
                    "rule_count": rule_count,
                    "use_llm": self.use_llm,
                },
            )
        return enriched_docs

    def transform_parents(
        self,
        parent_pairs: List[tuple],
        trace: Optional[IngestionTrace] = None,
    ) -> List[tuple]:
        """Enrich parent chunks: rule-based title; LLM summary + tags when enabled."""
        if not parent_pairs:
            return []

        if self.parent_use_llm and self.llm:
            return self._transform_parents_parallel(parent_pairs, trace)
        return self._transform_parents_sequential(parent_pairs, trace)

    def _enrich_parent_single(
        self,
        parent_id: str,
        document: Document,
    ) -> Tuple[tuple, str]:
        text = document.page_content or ""
        metadata = dict(document.metadata or {})
        metadata["title"] = self._extract_title(text)

        rule_metadata = self._rule_based_enrich(text)
        summary = rule_metadata["summary"]
        tags = rule_metadata["tags"]
        enriched_by = "rule"

        if self.parent_use_llm and self.llm:
            llm_metadata = self._llm_enrich_parent(text, metadata)
            if llm_metadata:
                summary = llm_metadata.get("summary") or summary
                tags = llm_metadata.get("tags") or tags
                enriched_by = "llm"

        metadata["summary"] = summary
        metadata["tags"] = self._normalize_tags(tags, config.INGESTION_PARENT_MAX_TAGS)
        metadata["enriched_by"] = enriched_by
        enriched = Document(page_content=text, metadata=metadata)
        return (parent_id, enriched), enriched_by

    def _transform_parents_parallel(
        self,
        parent_pairs: List[tuple],
        trace: Optional[IngestionTrace] = None,
    ) -> List[tuple]:
        max_workers = min(config.INGESTION_LLM_MAX_WORKERS, len(parent_pairs))
        results: List[Optional[tuple]] = [None] * len(parent_pairs)
        llm_count = 0
        rule_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(self._enrich_parent_single, parent_id, parent_doc): idx
                for idx, (parent_id, parent_doc) in enumerate(parent_pairs)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                parent_id, parent_doc = parent_pairs[idx]
                try:
                    pair, enriched_by = future.result()
                    results[idx] = pair
                    if enriched_by == "llm":
                        llm_count += 1
                    else:
                        rule_count += 1
                except Exception as exc:
                    logger.error("Parent metadata enrich failed for %s: %s", parent_id, exc)
                    metadata = dict(parent_doc.metadata or {})
                    rule_metadata = self._rule_based_enrich(parent_doc.page_content or "")
                    metadata["title"] = rule_metadata["title"]
                    metadata["summary"] = rule_metadata["summary"]
                    metadata["tags"] = self._normalize_tags(
                        rule_metadata["tags"],
                        config.INGESTION_PARENT_MAX_TAGS,
                    )
                    metadata["enriched_by"] = "rule"
                    results[idx] = (
                        parent_id,
                        Document(page_content=parent_doc.page_content, metadata=metadata),
                    )
                    rule_count += 1

        if trace:
            trace.record_stage(
                "parent_metadata_enricher",
                {
                    "total_parents": len(parent_pairs),
                    "llm_enhanced_count": llm_count,
                    "rule_count": rule_count,
                    "parent_use_llm": self.parent_use_llm,
                },
            )
        return [pair for pair in results if pair is not None]

    def _transform_parents_sequential(
        self,
        parent_pairs: List[tuple],
        trace: Optional[IngestionTrace] = None,
    ) -> List[tuple]:
        enriched_pairs = []
        llm_count = 0
        rule_count = 0

        for parent_id, parent_doc in parent_pairs:
            try:
                pair, enriched_by = self._enrich_parent_single(parent_id, parent_doc)
                enriched_pairs.append(pair)
                if enriched_by == "llm":
                    llm_count += 1
                else:
                    rule_count += 1
            except Exception as exc:
                logger.error("Parent metadata enrich failed for %s: %s", parent_id, exc)
                enriched_pairs.append((parent_id, parent_doc))
                rule_count += 1

        if trace:
            trace.record_stage(
                "parent_metadata_enricher",
                {
                    "total_parents": len(parent_pairs),
                    "llm_enhanced_count": llm_count,
                    "rule_count": rule_count,
                    "parent_use_llm": self.parent_use_llm,
                },
            )
        return enriched_pairs

    def _rule_based_enrich(self, text: str) -> Dict[str, Any]:
        return {
            "title": self._extract_title(text),
            "summary": self._extract_summary(text),
            "tags": self._extract_tags(text),
        }

    def _extract_title(self, text: str) -> str:
        if not text:
            return "Untitled"
        heading_match = re.match(r"^#{1,6}\s+(.+)$", text, re.MULTILINE)
        if heading_match:
            return heading_match.group(1).strip()
        first_line = text.split("\n", 1)[0].strip()
        if first_line and len(first_line) <= 100 and not first_line.endswith((".", ",", ";")):
            return first_line
        sentences = re.split(r"[.!?]\s+", text)
        if sentences and sentences[0]:
            title = re.sub(r"[.!?]+$", "", sentences[0].strip())
            return title[:147] + "..." if len(title) > 150 else title
        return text[:100].strip() + ("..." if len(text) > 100 else "")

    def _extract_summary(self, text: str, max_sentences: int = 3) -> str:
        if not text:
            return ""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        summary = " ".join(sentences[:max_sentences]).strip()
        return summary[:497] + "..." if len(summary) > 500 else summary

    def _extract_tags(self, text: str, max_tags: int = 10) -> List[str]:
        if not text:
            return []
        tags = set()
        tags.update(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)[:5])
        tags.update(re.findall(r"\b[a-z]+(?:[A-Z][a-z]*)+\b|\b[a-z]+_[a-z_]+\b", text)[:5])
        markdown_keywords = re.findall(r"\*\*(.+?)\*\*|\*(.+?)\*|__(.+?)__|_(.+?)_", text)
        for match in markdown_keywords[:5]:
            for group in match:
                if group:
                    tags.add(group.strip())
        return sorted(tags)[:max_tags]

    def _llm_enrich(self, text: str) -> Optional[Dict[str, Any]]:
        prompt_template = self._load_prompt()
        prompt = prompt_template.replace("{chunk_text}", text[:2000])
        response = invoke_text_llm(self.llm, prompt)
        if not response:
            return None
        return self._parse_llm_response(response)

    def _llm_enrich_parent(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        prompt_template = self._load_parent_prompt()
        section_context = self._build_section_context(metadata or {})
        prompt = (
            prompt_template.replace("{parent_text}", text[: config.INGESTION_PARENT_LLM_MAX_CHARS])
            .replace("{section_context}", section_context or "(none)")
        )
        response = invoke_text_llm(self.llm, prompt)
        if not response:
            return None
        return self._parse_parent_llm_response(response)

    @staticmethod
    def _build_section_context(metadata: Dict[str, Any]) -> str:
        parts = []
        for key in ("H1", "H2", "H3"):
            value = metadata.get(key)
            if value:
                parts.append(f"{key}: {value}")
        source = metadata.get("source")
        if source:
            parts.append(f"source: {source}")
        return "\n".join(parts)

    @staticmethod
    def _normalize_tags(tags: Any, max_tags: int) -> List[str]:
        if not tags:
            return []
        if isinstance(tags, str):
            candidates = [part.strip() for part in tags.split(",")]
        elif isinstance(tags, list):
            candidates = [str(tag).strip() for tag in tags]
        else:
            return []

        normalized: List[str] = []
        seen: set[str] = set()
        for tag in candidates:
            if not tag:
                continue
            key = tag.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(tag)
            if len(normalized) >= max_tags:
                break
        return normalized

    def _load_prompt(self) -> str:
        if self._prompt_template is None:
            self._prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
        return self._prompt_template

    def _load_parent_prompt(self) -> str:
        if self._parent_prompt_template is None:
            self._parent_prompt_template = PARENT_PROMPT_PATH.read_text(encoding="utf-8")
        return self._parent_prompt_template

    def _parse_parent_llm_response(self, response: str) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {"summary": "", "tags": []}
        summary_match = re.search(
            r"Summary:\s*(.+?)(?:\n(?:Tags:|$))",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        if summary_match:
            metadata["summary"] = summary_match.group(1).strip()
        tags_match = re.search(r"Tags:\s*(.+?)(?:\n|$)", response, re.IGNORECASE | re.DOTALL)
        if tags_match:
            metadata["tags"] = self._normalize_tags(
                tags_match.group(1),
                config.INGESTION_PARENT_MAX_TAGS,
            )
        if not metadata["summary"] and not metadata["tags"]:
            return {}
        if not metadata["summary"]:
            metadata["summary"] = response[:500].strip()
        metadata["tags"] = self._normalize_tags(
            metadata.get("tags", []),
            config.INGESTION_PARENT_MAX_TAGS,
        )
        return metadata

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        metadata = {"title": "", "summary": "", "tags": []}
        title_match = re.search(r"Title:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if title_match:
            metadata["title"] = title_match.group(1).strip()
        summary_match = re.search(
            r"Summary:\s*(.+?)(?:\n(?:Tags:|$))",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        if summary_match:
            metadata["summary"] = summary_match.group(1).strip()
        tags_match = re.search(r"Tags:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if tags_match:
            metadata["tags"] = self._normalize_tags(
                tags_match.group(1).split(","),
                config.INGESTION_PARENT_MAX_TAGS,
            )
        if not metadata["title"]:
            metadata["title"] = "Untitled"
        if not metadata["summary"]:
            metadata["summary"] = response[:200].strip()
        return metadata
