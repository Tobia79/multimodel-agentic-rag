"""Image captioning transform for chunks with [IMAGE: id] placeholders.

Supports OCR-only, VLM-only, and OCR-then-VLM (ocr_then_vlm) pipelines.
Skips images already processed at Load time (scanned PDF pages).
"""

from __future__ import annotations

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import config
from langchain_core.documents import Document

from ingestion.image_understanding import (
    ImageUnderstandingResult,
    effective_image_mode,
    understand_image,
)
from ingestion.llm_helpers import create_vision_llm
from ingestion.ocr import BaseOCR, create_ocr
from ingestion.trace import IngestionTrace
from ingestion.transform.base_transform import BaseTransform

logger = logging.getLogger(__name__)

IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\[IMAGE:\s*([^\]]+)\]")


@dataclass
class ImageProcessResult:
    ocr_text: Optional[str] = None
    caption: Optional[str] = None
    mode: str = "none"


class ImageCaptioner(BaseTransform):
    def __init__(self, llm=None, ocr: Optional[BaseOCR] = None):
        self._llm = llm
        self._ocr = ocr
        self._result_cache: Dict[str, ImageProcessResult] = {}
        self._cache_lock = threading.Lock()

    @property
    def llm(self):
        if self._llm is None and config.VISION_LLM_ENABLED:
            self._llm = create_vision_llm()
        return self._llm

    @property
    def ocr(self) -> Optional[BaseOCR]:
        if self._ocr is None and config.OCR_ENABLED:
            self._ocr = create_ocr()
        return self._ocr

    def transform(
        self,
        documents: List[Document],
        trace: Optional[IngestionTrace] = None,
    ) -> List[Document]:
        if not documents or not self._is_active():
            return documents

        image_lookup: Dict[str, dict] = {}
        for document in documents:
            for img_meta in (document.metadata or {}).get("images", []):
                img_id = img_meta.get("id")
                if img_id and img_id not in image_lookup:
                    image_lookup[img_id] = img_meta

        with self._cache_lock:
            self._result_cache.clear()

        images_to_process: Dict[str, str] = {}
        skipped_load_processed = 0
        for document in documents:
            for img_id in self._find_referenced_image_ids(document.page_content or ""):
                img_id_stripped = img_id.strip()
                if img_id_stripped in images_to_process:
                    continue
                img_meta = image_lookup.get(img_id_stripped)
                if not img_meta or not img_meta.get("path"):
                    continue
                if img_meta.get("load_processed"):
                    skipped_load_processed += 1
                    continue
                images_to_process[img_id_stripped] = img_meta["path"]

        if images_to_process:
            self._process_images_parallel(images_to_process)

        processed_docs = []
        enriched_chunks = 0
        total_ocr_chars = 0
        for document in documents:
            referenced_ids = self._find_referenced_image_ids(document.page_content or "")
            if not referenced_ids:
                processed_docs.append(document)
                continue

            new_text = document.page_content or ""
            captions_meta = []
            chunk_changed = False
            for img_id in referenced_ids:
                img_id_stripped = img_id.strip()
                img_meta = image_lookup.get(img_id_stripped) or {}

                if img_meta.get("load_processed"):
                    continue

                with self._cache_lock:
                    result = self._result_cache.get(img_id_stripped)
                if not result or not self._has_content(result):
                    continue

                captions_meta.append(
                    {
                        "id": img_id_stripped,
                        "ocr_text": result.ocr_text or "",
                        "caption": result.caption or "",
                        "mode": result.mode,
                    }
                )
                if result.ocr_text:
                    total_ocr_chars += len(result.ocr_text)

                placeholder = f"[IMAGE: {img_id}]"
                replacement = self._format_replacement(img_id, result)
                new_text = new_text.replace(placeholder, replacement)
                chunk_changed = True

            if chunk_changed:
                enriched_chunks += 1

            metadata = dict(document.metadata or {})
            if captions_meta:
                metadata["image_captions"] = captions_meta
            processed_docs.append(Document(page_content=new_text, metadata=metadata))

        if trace:
            trace.record_stage(
                "image_captioner",
                {
                    "enriched_chunks": enriched_chunks,
                    "captioned_chunks": enriched_chunks,
                    "unique_images": len(images_to_process),
                    "skipped_load_processed": skipped_load_processed,
                    "ocr_enabled": config.OCR_ENABLED,
                    "vision_enabled": config.VISION_LLM_ENABLED,
                    "mode": effective_image_mode(),
                    "ocr_chars_total": total_ocr_chars,
                },
            )
        return processed_docs

    def _is_active(self) -> bool:
        mode = effective_image_mode()
        if mode == "ocr_only":
            return config.OCR_ENABLED and self.ocr is not None
        if mode == "vlm_only":
            return config.VISION_LLM_ENABLED and self.llm is not None
        return (config.OCR_ENABLED and self.ocr is not None) or (
            config.VISION_LLM_ENABLED and self.llm is not None
        )

    def _find_referenced_image_ids(self, text: str) -> List[str]:
        return [match.strip() for match in IMAGE_PLACEHOLDER_PATTERN.findall(text)]

    def _process_images_parallel(self, images: Dict[str, str]) -> None:
        max_workers = min(config.INGESTION_VISION_MAX_WORKERS, len(images))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._process_image, img_id, img_path): img_id
                for img_id, img_path in images.items()
            }
            for future in as_completed(futures):
                img_id = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    logger.error("Image processing failed for %s: %s", img_id, exc)

    def _process_image(self, img_id: str, img_path: str) -> Optional[ImageProcessResult]:
        with self._cache_lock:
            if img_id in self._result_cache:
                return self._result_cache[img_id]

        understood: ImageUnderstandingResult = understand_image(
            img_path,
            purpose="caption",
            ocr=self.ocr,
            llm=self.llm,
        )
        result = ImageProcessResult(
            ocr_text=understood.ocr_text,
            caption=understood.structured_text,
            mode=understood.mode,
        )
        if self._has_content(result):
            with self._cache_lock:
                self._result_cache[img_id] = result
        return result

    @staticmethod
    def _has_content(result: ImageProcessResult) -> bool:
        return bool((result.ocr_text and result.ocr_text.strip()) or (result.caption and result.caption.strip()))

    @staticmethod
    def _format_replacement(img_id: str, result: ImageProcessResult) -> str:
        lines = [f"[IMAGE: {img_id}]"]
        if result.ocr_text and result.ocr_text.strip():
            lines.append(f"(OCR Text: {result.ocr_text.strip()})")
        if result.caption and result.caption.strip():
            lines.append(f"(Description: {result.caption.strip()})")
        return "\n".join(lines)
