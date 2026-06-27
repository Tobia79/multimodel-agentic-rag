"""Shared image understanding: OCR + optional VLM structuring."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import config
from ingestion.llm_helpers import create_vision_llm, invoke_vision_llm
from ingestion.ocr import BaseOCR, create_ocr

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
_VALID_MODES = frozenset({"ocr_only", "vlm_only", "ocr_then_vlm"})


@dataclass
class ImageUnderstandingResult:
    ocr_text: Optional[str] = None
    structured_text: Optional[str] = None
    mode: str = "none"


def _load_prompt(filename: str) -> str:
    path = _PROMPT_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return "Describe this image in detail for indexing purposes."


def effective_image_mode() -> str:
    mode = config.IMAGE_UNDERSTANDING_MODE
    if mode not in _VALID_MODES:
        logger.warning("Invalid IMAGE_UNDERSTANDING_MODE=%s, using ocr_then_vlm", mode)
        return "ocr_then_vlm"
    return mode


def understand_image(
    image_path: str,
    *,
    purpose: str = "caption",
    ocr: Optional[BaseOCR] = None,
    llm=None,
) -> ImageUnderstandingResult:
    """Run OCR / VLM pipeline on a single image file.

    purpose:
      - "caption": use image captioning prompts (Transform / ImageCaptioner)
      - "scan_page": use scanned page structuring prompt (Load / scanned PDF)
    """
    mode = effective_image_mode()
    ocr_engine = ocr if ocr is not None else create_ocr()
    vision_llm = llm
    if vision_llm is None and config.VISION_LLM_ENABLED:
        vision_llm = create_vision_llm()

    ocr_text: Optional[str] = None
    structured: Optional[str] = None
    result_mode = mode

    run_ocr = mode in {"ocr_only", "ocr_then_vlm"} and config.OCR_ENABLED and ocr_engine is not None
    run_vlm = mode in {"vlm_only", "ocr_then_vlm"} and config.VISION_LLM_ENABLED and vision_llm is not None

    if run_ocr:
        ocr_text = ocr_engine.extract_text(image_path)

    if run_vlm:
        if purpose == "scan_page":
            prompt = _load_prompt("scanned_page_structure.txt")
            if not ocr_text:
                prompt = _load_prompt("image_captioning.txt")
        elif ocr_text:
            prompt = _load_prompt("image_captioning_with_ocr.txt")
        else:
            prompt = _load_prompt("image_captioning.txt")
        structured = invoke_vision_llm(vision_llm, prompt, image_path, ocr_text=ocr_text)

    if mode == "ocr_then_vlm":
        if structured:
            result_mode = "ocr_then_vlm"
        elif ocr_text:
            result_mode = "ocr_only"
            structured = None
        else:
            result_mode = "none"
    elif mode == "ocr_only":
        result_mode = "ocr_only" if ocr_text else "none"
        structured = None
    elif mode == "vlm_only":
        result_mode = "vlm_only" if structured else "none"

    return ImageUnderstandingResult(
        ocr_text=ocr_text,
        structured_text=structured,
        mode=result_mode,
    )


def primary_text(result: ImageUnderstandingResult) -> str:
    """Best available text for indexing (structured VLM output, else OCR)."""
    if result.structured_text and result.structured_text.strip():
        return result.structured_text.strip()
    if result.ocr_text and result.ocr_text.strip():
        return result.ocr_text.strip()
    return ""
