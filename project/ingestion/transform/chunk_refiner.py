"""Chunk refinement: rule-based cleaning + optional LLM enhancement."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import config
from langchain_core.documents import Document

from ingestion.llm_helpers import create_ingestion_llm, invoke_text_llm
from ingestion.trace import IngestionTrace
from ingestion.transform.base_transform import BaseTransform

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "chunk_refinement.txt"


class ChunkRefiner(BaseTransform):
    def __init__(self, use_llm: Optional[bool] = None, llm=None):
        self.use_llm = (
            config.INGESTION_CHUNK_REFINER_USE_LLM if use_llm is None else use_llm
        )
        self._llm = llm
        self._prompt_template: Optional[str] = None

    @property
    def llm(self):
        if self.use_llm and self._llm is None:
            try:
                self._llm = create_ingestion_llm()
            except Exception as exc:
                logger.warning("ChunkRefiner LLM init failed: %s", exc)
                self.use_llm = False
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
        rule_refined = self._rule_based_refine(text)
        refined_by = "rule"
        refined_text = rule_refined

        if self.use_llm and self.llm:
            llm_refined = self._llm_refine(rule_refined)
            if llm_refined:
                refined_text = llm_refined
                refined_by = "llm"

        metadata = {**(document.metadata or {}), "refined_by": refined_by}
        return Document(page_content=refined_text, metadata=metadata), refined_by

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
                    refined_doc, refined_by = future.result()
                    results[idx] = refined_doc
                    if refined_by == "llm":
                        llm_count += 1
                    else:
                        rule_count += 1
                except Exception as exc:
                    logger.error("Chunk refine failed: %s", exc)
                    results[idx] = documents[idx]

        if trace:
            trace.record_stage(
                "chunk_refiner",
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
        refined_docs = []
        llm_count = 0
        rule_count = 0

        for document in documents:
            try:
                refined_doc, refined_by = self._transform_single(document, trace)
                refined_docs.append(refined_doc)
                if refined_by == "llm":
                    llm_count += 1
                else:
                    rule_count += 1
            except Exception as exc:
                logger.error("Chunk refine failed: %s", exc)
                refined_docs.append(document)

        if trace:
            trace.record_stage(
                "chunk_refiner",
                {
                    "total_chunks": len(documents),
                    "llm_enhanced_count": llm_count,
                    "rule_count": rule_count,
                    "use_llm": self.use_llm,
                },
            )
        return refined_docs

    def _rule_based_refine(self, text: str) -> str:
        if not text or not text.strip():
            return ""

        code_blocks: List[str] = []

        def extract_code_block(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

        text = re.sub(r"```[\s\S]*?```", extract_code_block, text)
        text = re.sub(
            r"(?:^|\n)--- end of page\.page_number=\d+ ---\s*",
            "\n",
            text,
            flags=re.MULTILINE,
        )
        text = re.sub(
            r"─{10,}.*?(?:Page \d+|Footer|Section \d+|©|Confidential).*?─{10,}",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        text = re.sub(r"─{10,}", "", text)
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = "\n".join(line.rstrip() for line in text.split("\n"))

        for index, code_block in enumerate(code_blocks):
            text = text.replace(f"__CODE_BLOCK_{index}__", code_block)
        return text.strip()

    def _llm_refine(self, text: str) -> Optional[str]:
        if not text.strip():
            return text
        prompt_template = self._load_prompt()
        if not prompt_template or "{text}" not in prompt_template:
            return None
        prompt = prompt_template.replace("{text}", text)
        return invoke_text_llm(self.llm, prompt)

    def _load_prompt(self) -> Optional[str]:
        if self._prompt_template is not None:
            return self._prompt_template
        if not PROMPT_PATH.exists():
            logger.warning("Chunk refinement prompt not found: %s", PROMPT_PATH)
            return None
        self._prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
        return self._prompt_template
