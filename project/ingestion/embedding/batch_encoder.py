"""Batch encoding for ingestion with trace-friendly statistics."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from langchain_core.documents import Document

from db.vector_db_manager import VectorDbManager


@dataclass
class EncodeResult:
    dense_vectors: List[List[float]]
    sparse_vectors: List[Any]
    dense_dimension: int
    dense_vector_count: int
    sparse_doc_count: int
    elapsed_ms: float
    chunk_details: List[Dict[str, Any]] = field(default_factory=list)


class BatchEncoder:
    def __init__(self, vector_db: VectorDbManager):
        self.vector_db = vector_db

    @staticmethod
    def _tokenize_for_sparse_display(text: str) -> List[str]:
        if not text:
            return []
        return re.findall(r"\b\w+\b", text.lower())

    @staticmethod
    def _sparse_pairs(sparse: Any) -> List[Dict[str, Any]]:
        values = getattr(sparse, "values", None)
        indices = getattr(sparse, "indices", None)
        if values is None or indices is None:
            return []
        pairs = [
            {"index": int(idx), "weight": float(weight)}
            for idx, weight in zip(indices, values)
        ]
        pairs.sort(key=lambda item: item["weight"], reverse=True)
        return pairs

    def encode(self, documents: List[Document]) -> EncodeResult:
        if not documents:
            return EncodeResult(
                dense_vectors=[],
                sparse_vectors=[],
                dense_dimension=0,
                dense_vector_count=0,
                sparse_doc_count=0,
                elapsed_ms=0.0,
            )

        texts = [doc.page_content for doc in documents]
        started = time.monotonic()
        dense_vectors, sparse_vectors = self.vector_db.embed_hybrid(texts)
        elapsed_ms = (time.monotonic() - started) * 1000.0

        dense_dimension = len(dense_vectors[0]) if dense_vectors else 0
        chunk_details = []
        for index, doc in enumerate(documents):
            input_text = doc.page_content or ""
            detail: Dict[str, Any] = {
                "chunk_index": index,
                "parent_id": (doc.metadata or {}).get("parent_id", ""),
                "char_len": len(input_text),
                "dense_dim": dense_dimension,
                "input_text": input_text,
                "metadata": {
                    k: v for k, v in dict(doc.metadata or {}).items() if k != "images"
                },
                "tokenized_terms": self._tokenize_for_sparse_display(input_text),
            }
            if index < len(sparse_vectors):
                sparse = sparse_vectors[index]
                pairs = self._sparse_pairs(sparse)
                detail["sparse_nonzero_terms"] = len(pairs)
                detail["sparse_pairs"] = pairs
            chunk_details.append(detail)

        return EncodeResult(
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
            dense_dimension=dense_dimension,
            dense_vector_count=len(dense_vectors),
            sparse_doc_count=len(sparse_vectors),
            elapsed_ms=elapsed_ms,
            chunk_details=chunk_details,
        )
