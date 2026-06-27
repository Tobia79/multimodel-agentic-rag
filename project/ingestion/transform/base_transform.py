"""Base transform interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_core.documents import Document

from ingestion.trace import IngestionTrace


class BaseTransform(ABC):
    @abstractmethod
    def transform(
        self,
        documents: List[Document],
        trace: Optional[IngestionTrace] = None,
    ) -> List[Document]:
        raise NotImplementedError
