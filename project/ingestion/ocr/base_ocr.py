"""Base OCR interface for image text extraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class BaseOCR(ABC):
    @abstractmethod
    def extract_text(self, image_path: str) -> Optional[str]:
        """Extract visible text from an image file. Returns None on failure."""
        raise NotImplementedError
