"""Dense + sparse retrieval with application-layer RRF fusion."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Tuple

import config
from core.fusion import RRFFusion, RankedDocument
from langchain_qdrant import QdrantVectorStore, RetrievalMode

logger = logging.getLogger(__name__)


def document_chunk_id(document: Any) -> str:
    metadata = getattr(document, "metadata", None) or {}
    point_id = metadata.get("_id")
    if point_id is not None:
        return str(point_id)
    parent_id = metadata.get("parent_id", "unknown")
    content = getattr(document, "page_content", "")
    return f"{parent_id}:{hash(content)}"


def _to_ranked(documents_with_scores: List[Tuple[Any, float]]) -> List[RankedDocument]:
    ranked: List[RankedDocument] = []
    for document, score in documents_with_scores:
        ranked.append(
            RankedDocument(
                chunk_id=document_chunk_id(document),
                score=float(score),
                document=document,
            )
        )
    return ranked


class HybridSearchEngine:
    """Run dense and sparse retrieval separately, then fuse with RRF."""

    def __init__(self, collection: QdrantVectorStore) -> None:
        self._collection = collection
        self._fusion = RRFFusion()
        self._dense_store = self._store_for_mode(RetrievalMode.DENSE)
        self._sparse_store = self._store_for_mode(RetrievalMode.SPARSE)

    def _store_for_mode(self, mode: RetrievalMode) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=self._collection.client,
            collection_name=self._collection.collection_name,
            embedding=self._collection.embeddings,
            sparse_embedding=self._collection.sparse_embeddings,
            retrieval_mode=mode,
            sparse_vector_name=config.SPARSE_VECTOR_NAME,
            validate_collection_config=False,
        )

    def _dense_search(self, query: str, top_k: int) -> Tuple[List[RankedDocument], Optional[str]]:
        try:
            results = self._dense_store.similarity_search_with_score(query, k=top_k)
            return _to_ranked(results), None
        except Exception as exc:
            logger.warning("Dense retrieval failed: %s", exc)
            return [], str(exc)

    def _sparse_search(self, query: str, top_k: int) -> Tuple[List[RankedDocument], Optional[str]]:
        try:
            results = self._sparse_store.similarity_search_with_score(query, k=top_k)
            return _to_ranked(results), None
        except Exception as exc:
            logger.warning("Sparse retrieval failed: %s", exc)
            return [], str(exc)

    def search(
        self,
        query: str,
        fusion_top_k: int,
        *,
        dense_top_k: Optional[int] = None,
        sparse_top_k: Optional[int] = None,
    ) -> List[Any]:
        """Return LangChain documents after dense + sparse + RRF."""
        if not query or not query.strip():
            return []
        if fusion_top_k < 1:
            return []

        dense_k = dense_top_k if dense_top_k is not None else config.DENSE_TOP_K
        sparse_k = sparse_top_k if sparse_top_k is not None else config.SPARSE_TOP_K

        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(self._dense_search, query, dense_k)
            sparse_future = executor.submit(self._sparse_search, query, sparse_k)
            dense_results, dense_error = dense_future.result()
            sparse_results, sparse_error = sparse_future.result()

        if dense_error and sparse_error:
            logger.error(
                "Both retrieval paths failed. dense=%s sparse=%s",
                dense_error,
                sparse_error,
            )
            return []

        if dense_error:
            logger.warning("Dense failed, using sparse only: %s", dense_error)
            fused = sparse_results[:fusion_top_k]
        elif sparse_error:
            logger.warning("Sparse failed, using dense only: %s", sparse_error)
            fused = dense_results[:fusion_top_k]
        elif not dense_results and not sparse_results:
            fused = []
        else:
            ranking_lists = [lst for lst in (dense_results, sparse_results) if lst]
            if len(ranking_lists) == 1:
                fused = ranking_lists[0][:fusion_top_k]
            else:
                fused = self._fusion.fuse(ranking_lists, top_k=fusion_top_k)

        return [item.document for item in fused]
