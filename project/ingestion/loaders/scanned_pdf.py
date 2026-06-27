"""Scanned PDF processing: per-page render → OCR → VLM structuring."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import config
from ingestion.image_understanding import ImageUnderstandingResult, primary_text, understand_image

logger = logging.getLogger(__name__)

try:
    import pymupdf

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

_PAGE_SEPARATOR_TEMPLATE = "\n--- end of page.page_number={page} ---\n"


@dataclass
class _PageWorkItem:
    page_num: int
    image_path: str
    image_meta: Dict[str, Any]


def detect_scanned_pdf(doc, markdown_text: str) -> bool:
    """Heuristic: few extractable text characters per page → treat as scanned."""
    mode = config.INGESTION_PDF_SCAN_MODE
    if mode == "never":
        return False
    if mode == "always":
        return True

    page_count = len(doc)
    if page_count == 0:
        return False

    layer_chars = 0
    for page_index in range(page_count):
        layer_chars += len((doc[page_index].get_text() or "").strip())

    md_stripped = re.sub(r"\s+", "", markdown_text or "")
    avg_layer = layer_chars / page_count
    avg_md = len(md_stripped) / page_count
    threshold = config.INGESTION_PDF_SCAN_TEXT_THRESHOLD

    is_scanned = avg_layer < threshold and avg_md < threshold * 2
    if is_scanned:
        logger.info(
            "Detected scanned PDF (%d pages, avg text layer chars=%.1f, threshold=%d)",
            page_count,
            avg_layer,
            threshold,
        )
    return is_scanned


def is_scanned_pdf_processing_enabled() -> bool:
    if not config.INGESTION_PDF_SCAN_OCR:
        return False
    if not config.OCR_ENABLED:
        logger.warning("INGESTION_PDF_SCAN_OCR requires OCR_ENABLED=true")
        return False
    return True


class ScannedPdfProcessor:
    """Render each PDF page to an image and run OCR → VLM at load time."""

    def __init__(
        self,
        *,
        doc_hash: str,
        store_image: Callable[..., Dict[str, Any]],
    ):
        self.doc_hash = doc_hash
        self._store_image = store_image

    def process(self, doc) -> Tuple[str, List[Dict[str, Any]]]:
        if not PYMUPDF_AVAILABLE:
            raise RuntimeError("pymupdf is required for scanned PDF processing")

        page_count = len(doc)
        zoom = config.INGESTION_PDF_SCAN_DPI / 72.0
        render_matrix = pymupdf.Matrix(zoom, zoom)

        work_items: List[_PageWorkItem] = []
        images_metadata: List[Dict[str, Any]] = []

        for page_index in range(page_count):
            page_num = page_index + 1
            page = doc[page_index]
            try:
                pixmap = page.get_pixmap(matrix=render_matrix, alpha=False)
                image_bytes = pixmap.tobytes("png")
            except Exception as exc:
                logger.warning("Failed to render scanned page %s: %s", page_num, exc)
                continue

            meta = self._store_image(
                doc_hash=self.doc_hash,
                image_index=page_num,
                image_bytes=image_bytes,
                image_ext="png",
                page=page_num,
                id_suffix=f"p{page_num}",
                source_ref="scan_page_render",
            )
            meta["role"] = "scan_page"
            meta["load_processed"] = False
            images_metadata.append(meta)
            work_items.append(
                _PageWorkItem(
                    page_num=page_num,
                    image_path=meta["path"],
                    image_meta=meta,
                )
            )

        page_bodies: Dict[int, str] = {}
        if work_items:
            page_bodies = self._process_pages_parallel(work_items)

        for item in work_items:
            item.image_meta["load_processed"] = item.page_num in page_bodies

        return self._join_pages(page_count, page_bodies, work_items), images_metadata

    def _process_pages_parallel(self, work_items: List[_PageWorkItem]) -> Dict[int, str]:
        bodies: Dict[int, str] = {}
        max_workers = min(config.INGESTION_VISION_MAX_WORKERS, len(work_items))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._process_single_page, item): item.page_num
                for item in work_items
            }
            for future in as_completed(futures):
                page_num = futures[future]
                try:
                    body = future.result()
                    if body:
                        bodies[page_num] = body
                except Exception as exc:
                    logger.error("Scanned page processing failed for page %s: %s", page_num, exc)
        return bodies

    def _process_single_page(self, item: _PageWorkItem) -> str:
        result: ImageUnderstandingResult = understand_image(
            item.image_path,
            purpose="scan_page",
        )
        body = primary_text(result)
        if not body:
            logger.warning("No text produced for scanned page %s", item.page_num)
            return ""

        item.image_meta["ocr_text"] = result.ocr_text or ""
        item.image_meta["structured_text"] = result.structured_text or ""
        item.image_meta["understanding_mode"] = result.mode

        image_id = item.image_meta["id"]
        return f"{body}\n\n[IMAGE: {image_id}]"

    @staticmethod
    def _join_pages(
        page_count: int,
        page_bodies: Dict[int, str],
        work_items: List[_PageWorkItem],
    ) -> str:
        image_id_by_page = {item.page_num: item.image_meta["id"] for item in work_items}
        chunks: List[str] = []
        for page_num in range(1, page_count + 1):
            body = page_bodies.get(page_num, "")
            if not body and page_num in image_id_by_page:
                body = f"[IMAGE: {image_id_by_page[page_num]}]"
            chunks.append(body.rstrip("\n"))
            if page_num < page_count:
                chunks.append(_PAGE_SEPARATOR_TEMPLATE.format(page=page_num))
        return "".join(chunks)
