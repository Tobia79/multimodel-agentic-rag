"""PaddleOCR backend for image text extraction."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, List, Optional

import config

from ingestion.ocr.base_ocr import BaseOCR

logger = logging.getLogger(__name__)

# Windows CPU: Paddle 3.3 + oneDNN can crash without these (PaddleOCR #17539)
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def _parse_predict_result(result: Any) -> List[str]:
    texts: List[str] = []
    if not result:
        return texts

    for item in result:
        if isinstance(item, dict):
            rec_texts = item.get("rec_texts") or item.get("text") or []
            if isinstance(rec_texts, list):
                texts.extend(str(t).strip() for t in rec_texts if t and str(t).strip())
            elif rec_texts:
                text = str(rec_texts).strip()
                if text:
                    texts.append(text)
        elif isinstance(item, (list, tuple)):
            for line in item:
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    text_part = line[1]
                    if isinstance(text_part, (list, tuple)) and text_part:
                        text = str(text_part[0]).strip()
                        if text:
                            texts.append(text)
                    elif isinstance(text_part, str):
                        text = text_part.strip()
                        if text:
                            texts.append(text)
    return texts


class PaddleOCREngine(BaseOCR):
    """Thread-safe lazy singleton around PaddleOCR predict()."""

    def __init__(self) -> None:
        self._engine = None
        self._init_lock = threading.Lock()
        self._predict_lock = threading.Lock()

    def _get_engine(self):
        if self._engine is not None:
            return self._engine

        with self._init_lock:
            if self._engine is not None:
                return self._engine

            from paddleocr import PaddleOCR

            lang = config.OCR_LANG if config.OCR_LANG in {"ch", "en", "japan", "korean"} else "ch"
            logger.info("Initializing PaddleOCR lang=%s enable_mkldnn=False", lang)
            self._engine = PaddleOCR(
                use_textline_orientation=True,
                lang=lang,
                enable_mkldnn=False,
            )
            return self._engine

    def extract_text(self, image_path: str) -> Optional[str]:
        try:
            with self._predict_lock:
                result = self._get_engine().predict(image_path)
            lines = _parse_predict_result(result)
            if not lines:
                return None
            return "\n".join(lines)
        except Exception as exc:
            logger.warning("PaddleOCR failed for %s: %s", image_path, exc)
            return None
