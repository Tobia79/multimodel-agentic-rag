"""OCR backends for ingestion image processing."""

from __future__ import annotations

import logging
from typing import Optional

import config

from ingestion.ocr.base_ocr import BaseOCR

logger = logging.getLogger(__name__)

_OCR_SINGLETON: Optional[BaseOCR] = None


def create_ocr() -> Optional[BaseOCR]:
    """Create or return cached OCR engine based on config.OCR_PROVIDER."""
    global _OCR_SINGLETON

    if not config.OCR_ENABLED:
        return None

    if _OCR_SINGLETON is not None:
        return _OCR_SINGLETON

    provider = config.OCR_PROVIDER
    if provider == "paddle":
        try:
            from ingestion.ocr.paddle_ocr import PaddleOCREngine

            _OCR_SINGLETON = PaddleOCREngine()
            return _OCR_SINGLETON
        except ImportError:
            logger.error(
                "OCR_ENABLED but paddleocr is not installed. "
                "Run: pip install paddlepaddle paddleocr"
            )
            return None

    logger.warning("Unsupported OCR_PROVIDER: %s", provider)
    return None


__all__ = ["BaseOCR", "create_ocr"]
