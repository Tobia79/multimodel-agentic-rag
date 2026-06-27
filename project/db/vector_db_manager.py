import uuid
import config
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

class VectorDbManager:
    DENSE_VECTOR_NAME = ""
    CONTENT_PAYLOAD_KEY = "page_content"
    METADATA_PAYLOAD_KEY = "metadata"

    __client: QdrantClient
    __dense_embeddings: HuggingFaceEmbeddings
    __sparse_embeddings: FastEmbedSparse
    def __init__(self):
        self.__client = QdrantClient(path=config.QDRANT_DB_PATH)
        self.__dense_embeddings = HuggingFaceEmbeddings(
            model_name=config.DENSE_MODEL,
            model_kwargs={"local_files_only": True},
        )
        self.__sparse_embeddings = FastEmbedSparse(
            model_name=config.SPARSE_MODEL,
            specific_model_path=config.SPARSE_MODEL_PATH,
            local_files_only=True,
        )

    def _dense_vector_size(self):
        return len(self.__dense_embeddings.embed_query("test"))

    def embed_hybrid(self, texts: List[str]) -> Tuple[List[List[float]], List[Any]]:
        dense_vectors = self.__dense_embeddings.embed_documents(texts)
        sparse_vectors = self.__sparse_embeddings.embed_documents(texts)
        if len(dense_vectors) != len(sparse_vectors):
            raise ValueError("Dense and sparse embedding counts do not match")
        return dense_vectors, sparse_vectors

    def upsert_hybrid_documents(
        self,
        collection_name: str,
        documents: List[Document],
        dense_vectors: List[List[float]],
        sparse_vectors: List[Any],
    ) -> List[str]:
        if not documents:
            return []
        if len(documents) != len(dense_vectors) or len(documents) != len(sparse_vectors):
            raise ValueError("Document and vector counts must match")

        point_ids: List[str] = []
        points: List[qmodels.PointStruct] = []
        for doc, dense_vector, sparse_vector in zip(documents, dense_vectors, sparse_vectors):
            point_id = uuid.uuid4().hex
            point_ids.append(point_id)
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector={
                        self.DENSE_VECTOR_NAME: dense_vector,
                        config.SPARSE_VECTOR_NAME: qmodels.SparseVector(
                            values=sparse_vector.values,
                            indices=sparse_vector.indices,
                        ),
                    },
                    payload={
                        self.CONTENT_PAYLOAD_KEY: doc.page_content,
                        self.METADATA_PAYLOAD_KEY: doc.metadata or {},
                    },
                )
            )

        self.__client.upsert(collection_name=collection_name, points=points)
        return point_ids

    def count_chunks_by_doc_hash(self, collection_name: str, doc_hash: str) -> int:
        return len(self._scroll_point_ids(collection_name, doc_hash=doc_hash))

    def count_chunks_by_source(self, collection_name: str, source_name: str) -> int:
        return len(self._scroll_point_ids(collection_name, source_name=source_name))

    def delete_by_doc_hash(self, collection_name: str, doc_hash: str) -> int:
        point_ids = self._scroll_point_ids(collection_name, doc_hash=doc_hash)
        if not point_ids:
            return 0
        self.__client.delete(
            collection_name=collection_name,
            points_selector=qmodels.PointIdsList(points=point_ids),
        )
        return len(point_ids)

    def delete_by_source(self, collection_name: str, source_name: str) -> int:
        point_ids = self._scroll_point_ids(collection_name, source_name=source_name)
        if not point_ids:
            return 0
        self.__client.delete(
            collection_name=collection_name,
            points_selector=qmodels.PointIdsList(points=point_ids),
        )
        return len(point_ids)

    def _scroll_point_ids(
        self,
        collection_name: str,
        *,
        doc_hash: Optional[str] = None,
        source_name: Optional[str] = None,
    ) -> List[str]:
        if doc_hash is None and source_name is None:
            return []

        conditions = []
        if doc_hash is not None:
            conditions.append(
                qmodels.FieldCondition(
                    key=f"{self.METADATA_PAYLOAD_KEY}.doc_hash",
                    match=qmodels.MatchValue(value=doc_hash),
                )
            )
        if source_name is not None:
            conditions.append(
                qmodels.FieldCondition(
                    key=f"{self.METADATA_PAYLOAD_KEY}.source",
                    match=qmodels.MatchValue(value=source_name),
                )
            )

        scroll_filter = qmodels.Filter(must=conditions)
        point_ids: List[str] = []
        offset = None
        while True:
            records, offset = self.__client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=256,
                offset=offset,
                with_payload=False,
            )
            point_ids.extend(str(record.id) for record in records)
            if offset is None:
                break
        return point_ids

    @staticmethod
    def _collection_vector_size(collection_info):
        vectors_config = collection_info.config.params.vectors
        if hasattr(vectors_config, "size"):
            return vectors_config.size
        if isinstance(vectors_config, dict) and vectors_config:
            first_vector = next(iter(vectors_config.values()))
            return getattr(first_vector, "size", None)
        return None

    def create_collection(self, collection_name):
        expected_size = self._dense_vector_size()
        if not self.__client.collection_exists(collection_name):
            print(f"Creating collection: {collection_name}...")
            self.__client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(size=expected_size, distance=qmodels.Distance.COSINE),
                sparse_vectors_config={config.SPARSE_VECTOR_NAME: qmodels.SparseVectorParams()},
            )
            print(f"✓ Collection created: {collection_name}")
        else:
            collection_info = self.__client.get_collection(collection_name)
            existing_size = self._collection_vector_size(collection_info)
            if existing_size and existing_size != expected_size:
                raise ValueError(
                    f"Qdrant collection '{collection_name}' has dense vector size "
                    f"{existing_size}, but '{config.DENSE_MODEL}' produces size "
                    f"{expected_size}. Clear and re-index the collection after "
                    "changing embedding models."
                )
            print(f"✓ Collection already exists: {collection_name}")

    def delete_collection(self, collection_name):
        try:
            if self.__client.collection_exists(collection_name):
                print(f"Removing existing Qdrant collection: {collection_name}")
                self.__client.delete_collection(collection_name)
        except Exception as e:
            print(f"Warning: could not delete collection {collection_name}: {e}")

    def get_collection(self, collection_name) -> QdrantVectorStore:
        try:
            return QdrantVectorStore(
                    client=self.__client,
                    collection_name=collection_name,
                    embedding=self.__dense_embeddings,
                    sparse_embedding=self.__sparse_embeddings,
                    retrieval_mode=RetrievalMode.HYBRID,
                    sparse_vector_name=config.SPARSE_VECTOR_NAME
                )
        except Exception as e:
            print(f"Unable to get collection {collection_name}: {e}")