"""Ingestion pipeline orchestrator."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import config
from langchain_core.documents import Document

from document_chunker import DocumentChuncker
from ingestion.file_integrity import FileIntegrityChecker
from ingestion.loaders.document_loader import LoadedDocument, UniversalDocumentLoader
from ingestion.trace import IngestionTrace, IngestionTraceCollector
from ingestion.transform.chunk_refiner import ChunkRefiner
from ingestion.transform.image_captioner import ImageCaptioner
from ingestion.transform.metadata_enricher import MetadataEnricher
from ingestion.embedding.batch_encoder import BatchEncoder

logger = logging.getLogger(__name__)

_IMAGE_REF_PATTERN = re.compile(r"\[IMAGE:\s*([^\]]+)\]")


@dataclass
class PipelineResult:
    success: bool
    file_path: str
    doc_id: Optional[str] = None
    parent_count: int = 0
    child_count: int = 0
    image_count: int = 0
    error: Optional[str] = None
    skipped: bool = False
    stages: Dict[str, Any] = field(default_factory=dict)


class IngestionPipeline:
    STAGE_NAMES = ("integrity", "load", "split", "transform", "embed", "upsert")

    def __init__(
        self,
        rag_system,
        collection: str = "default",
        force: bool = False,
    ):
        self.rag_system = rag_system
        self.collection = collection
        self.force = force

        self.integrity_checker = FileIntegrityChecker(config.INGESTION_DB_PATH)
        self.loader = UniversalDocumentLoader(extract_images=config.INGESTION_EXTRACT_IMAGES)
        self.chunker = rag_system.chunker
        self.chunk_refiner = ChunkRefiner()
        self.metadata_enricher = MetadataEnricher()
        self.image_captioner = ImageCaptioner()
        self.batch_encoder = BatchEncoder(rag_system.vector_db)
        self.trace_collector = IngestionTraceCollector(config.INGESTION_TRACE_FILE)

    def run(
        self,
        file_path: str,
        trace: Optional[IngestionTrace] = None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> PipelineResult:
        path = Path(file_path)
        stages: Dict[str, Any] = {}
        total_stages = len(self.STAGE_NAMES)
        file_hash = ""

        def notify(stage_name: str, step: int) -> None:
            if on_progress is not None:
                on_progress(stage_name, step, total_stages)

        try:
            notify("integrity", 1)
            file_hash = self.integrity_checker.compute_sha256(str(path))
            if not self.force and self.integrity_checker.should_skip(file_hash):
                logger.info("Skipping already processed file: %s", path)
                return PipelineResult(
                    success=True,
                    file_path=str(path),
                    doc_id=file_hash,
                    skipped=True,
                    stages={"integrity": {"skipped": True}},
                )
            stages["integrity"] = {"file_hash": file_hash, "skipped": False}

            notify("load", 2)
            started = time.monotonic()
            loaded = self._load_document(path, file_hash=file_hash)
            load_elapsed = (time.monotonic() - started) * 1000.0
            image_count = len(loaded.metadata.get("images", []))
            stages["loading"] = {
                "text_length": len(loaded.text),
                "image_count": image_count,
                "doc_type": loaded.metadata.get("doc_type"),
            }
            if trace is not None:
                trace.record_stage(
                    "load",
                    {
                        "text_length": len(loaded.text),
                        "image_count": image_count,
                        "md_path": str(loaded.md_path) if loaded.md_path else None,
                        "doc_type": loaded.metadata.get("doc_type"),
                        "doc_hash": file_hash,
                        "title": loaded.metadata.get("title"),
                        "text_full": loaded.text,
                        "images": loaded.metadata.get("images", []),
                    },
                    elapsed_ms=load_elapsed,
                )

            notify("split", 3)
            started = time.monotonic()
            parent_pairs, child_chunks = self.chunker.create_chunks_single(loaded.md_path)
            split_elapsed = (time.monotonic() - started) * 1000.0
            if not child_chunks:
                raise ValueError(f"No child chunks generated for {path.name}")

            child_chunks = self._attach_document_metadata(child_chunks, loaded)
            parent_pairs = self._attach_parent_metadata(parent_pairs, loaded)
            stages["chunking"] = {
                "parent_count": len(parent_pairs),
                "child_count": len(child_chunks),
            }
            if trace is not None:
                avg_chunk_size = (
                    sum(len(c.page_content) for c in child_chunks) // len(child_chunks)
                    if child_chunks
                    else 0
                )
                trace.record_stage(
                    "split",
                    {
                        "parent_count": len(parent_pairs),
                        "child_count": len(child_chunks),
                        "chunk_count": len(child_chunks),
                        "avg_chunk_size": avg_chunk_size,
                        "parents": [
                            {
                                "parent_index": index,
                                "parent_id": parent_id,
                                "char_len": len(parent_doc.page_content),
                                "text": parent_doc.page_content,
                                "metadata": dict(parent_doc.metadata or {}),
                            }
                            for index, (parent_id, parent_doc) in enumerate(parent_pairs)
                        ],
                        "chunks": [
                            self._child_chunk_trace_payload(index, chunk)
                            for index, chunk in enumerate(child_chunks)
                        ],
                    },
                    elapsed_ms=split_elapsed,
                )

            notify("transform", 4)
            started = time.monotonic()
            pre_refine_texts = {
                index: chunk.page_content for index, chunk in enumerate(child_chunks)
            }
            child_chunks = self.chunk_refiner.transform(child_chunks, trace)
            post_refine_texts = {
                index: chunk.page_content for index, chunk in enumerate(child_chunks)
            }
            child_chunks = self.metadata_enricher.transform(child_chunks, trace)
            post_enrich_texts = {
                index: chunk.page_content for index, chunk in enumerate(child_chunks)
            }
            child_chunks = self.image_captioner.transform(child_chunks, trace)
            parent_pairs = self._transform_parents(parent_pairs, trace=trace)
            transform_elapsed = (time.monotonic() - started) * 1000.0
            stages["transform"] = {
                "refined_by_llm": sum(
                    1 for chunk in child_chunks if chunk.metadata.get("refined_by") == "llm"
                ),
                "refined_by_rule": sum(
                    1 for chunk in child_chunks if chunk.metadata.get("refined_by") == "rule"
                ),
                "enriched_by_llm": sum(
                    1 for chunk in child_chunks if chunk.metadata.get("enriched_by") == "llm"
                ),
                "enriched_by_rule": sum(
                    1 for chunk in child_chunks if chunk.metadata.get("enriched_by") == "rule"
                ),
                "captioned_chunks": sum(
                    1 for chunk in child_chunks if chunk.metadata.get("image_captions")
                ),
            }
            if trace is not None:
                transform_payload = {
                    **stages["transform"],
                    "chunks": [
                        {
                            "chunk_index": index,
                            "char_len": len(chunk.page_content),
                            "parent_id": (chunk.metadata or {}).get("parent_id", ""),
                            "text_before": pre_refine_texts.get(index, ""),
                            "text_after_refine": post_refine_texts.get(index, ""),
                            "text_after_enrich": post_enrich_texts.get(index, ""),
                            "text_after": chunk.page_content,
                            "refined_by": (chunk.metadata or {}).get("refined_by", ""),
                            "enriched_by": (chunk.metadata or {}).get("enriched_by", ""),
                            "title": (chunk.metadata or {}).get("title", ""),
                            "summary": (chunk.metadata or {}).get("summary", ""),
                            "tags": (chunk.metadata or {}).get("tags", []),
                            "metadata": {
                                k: v
                                for k, v in dict(chunk.metadata or {}).items()
                                if k != "images"
                            },
                            "image_captions": (chunk.metadata or {}).get("image_captions", []),
                        }
                        for index, chunk in enumerate(child_chunks)
                    ],
                }
                trace.record_stage(
                    "transform",
                    transform_payload,
                    elapsed_ms=transform_elapsed,
                )

            notify("embed", 5)
            started = time.monotonic()
            encode_result = self.batch_encoder.encode(child_chunks)
            embed_elapsed = (time.monotonic() - started) * 1000.0
            stages["encoding"] = {
                "dense_vector_count": encode_result.dense_vector_count,
                "dense_dimension": encode_result.dense_dimension,
                "sparse_doc_count": encode_result.sparse_doc_count,
            }
            if trace is not None:
                trace.record_stage(
                    "embed",
                    {
                        "method": "hybrid",
                        "dense_model": config.DENSE_MODEL,
                        "sparse_model": config.SPARSE_MODEL,
                        "dense_vector_count": encode_result.dense_vector_count,
                        "dense_dimension": encode_result.dense_dimension,
                        "sparse_doc_count": encode_result.sparse_doc_count,
                        "chunks": encode_result.chunk_details,
                    },
                    elapsed_ms=embed_elapsed,
                )

            notify("upsert", 6)
            started = time.monotonic()
            vector_ids = self.rag_system.vector_db.upsert_hybrid_documents(
                self.rag_system.collection_name,
                child_chunks,
                encode_result.dense_vectors,
                encode_result.sparse_vectors,
            )
            self.rag_system.parent_store.save_many(parent_pairs)
            upsert_elapsed = (time.monotonic() - started) * 1000.0
            stages["storage"] = {
                "child_count": len(child_chunks),
                "parent_count": len(parent_pairs),
                "vector_ids_count": len(vector_ids),
            }
            if trace is not None:
                trace.record_stage(
                    "upsert",
                    {
                        "child_count": len(child_chunks),
                        "parent_count": len(parent_pairs),
                        "vector_count": len(vector_ids),
                        "vector_ids": vector_ids,
                        "dense_store": {
                            "backend": "Qdrant",
                            "collection": self.rag_system.collection_name,
                            "count": len(vector_ids),
                            "path": config.QDRANT_DB_PATH,
                        },
                        "parent_store": {
                            "backend": "JSON",
                            "count": len(parent_pairs),
                            "path": config.PARENT_STORE_PATH,
                        },
                        "image_store": {
                            "count": image_count,
                            "path": config.INGESTION_IMAGES_DIR,
                        },
                    },
                    elapsed_ms=upsert_elapsed,
                )

            self.integrity_checker.mark_success(file_hash, str(path), self.collection)
            return PipelineResult(
                success=True,
                file_path=str(path),
                doc_id=file_hash,
                parent_count=len(parent_pairs),
                child_count=len(child_chunks),
                image_count=image_count,
                stages=stages,
            )
        except Exception as exc:
            logger.exception("Ingestion pipeline failed for %s", path)
            if file_hash:
                self.integrity_checker.mark_failed(file_hash, str(path), str(exc))
            return PipelineResult(
                success=False,
                file_path=str(path),
                doc_id=file_hash or None,
                error=str(exc),
                stages=stages,
            )

    @staticmethod
    def _child_chunk_trace_payload(index: int, chunk: Document) -> Dict[str, Any]:
        text = chunk.page_content or ""
        metadata = dict(chunk.metadata or {})
        doc_images = metadata.pop("images", None) or []
        refs = [ref.strip() for ref in _IMAGE_REF_PATTERN.findall(text)]
        ref_set = set(refs)
        images_in_chunk = [img for img in doc_images if img.get("id") in ref_set]

        body_lines: List[str] = []
        image_lines: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if _IMAGE_REF_PATTERN.search(line) or stripped.startswith("(Description:"):
                image_lines.append(line)
            else:
                body_lines.append(line)

        return {
            "chunk_index": index,
            "char_len": len(text),
            "parent_id": metadata.get("parent_id", ""),
            "text": text,
            "text_body": "\n".join(body_lines).strip(),
            "image_lines": "\n".join(image_lines).strip(),
            "image_refs": refs,
            "images_in_chunk": images_in_chunk,
            "metadata": metadata,
        }

    def _load_document(self, path: Path, file_hash: str) -> LoadedDocument:
        suffix = path.suffix.lower()
        if suffix in {".md", ".markdown"}:
            loaded = self.loader.copy_markdown_file(path, file_hash=file_hash)
        else:
            loaded = self.loader.load(path, file_hash=file_hash)
        if not loaded.md_path or not loaded.md_path.exists():
            raise RuntimeError(f"Failed to persist markdown for {path.name}")
        if loaded.file_hash != file_hash:
            raise RuntimeError(
                f"Loaded document hash mismatch for {path.name}: "
                f"expected {file_hash}, got {loaded.file_hash}"
            )
        return loaded

    @staticmethod
    def _attach_document_metadata(
        child_chunks: List[Document],
        loaded: LoadedDocument,
    ) -> List[Document]:
        images = loaded.metadata.get("images", [])
        doc_hash = loaded.file_hash
        updated = []
        for chunk in child_chunks:
            metadata = dict(chunk.metadata or {})
            metadata["doc_hash"] = doc_hash
            if images:
                metadata["images"] = images
            updated.append(Document(page_content=chunk.page_content, metadata=metadata))
        return updated

    @staticmethod
    def _attach_parent_metadata(
        parent_pairs: List[tuple],
        loaded: LoadedDocument,
    ) -> List[tuple]:
        updated = []
        for parent_id, parent_doc in parent_pairs:
            metadata = dict(parent_doc.metadata or {})
            metadata["doc_hash"] = loaded.file_hash
            updated.append(
                (
                    parent_id,
                    Document(page_content=parent_doc.page_content, metadata=metadata),
                )
            )
        return updated

    def _transform_parents(
        self,
        parent_pairs: List[tuple],
        trace: Optional[IngestionTrace] = None,
    ) -> List[tuple]:
        cleaned_pairs = []
        for parent_id, parent_doc in parent_pairs:
            cleaned = self.chunk_refiner._rule_based_refine(parent_doc.page_content or "")
            metadata = dict(parent_doc.metadata or {})
            metadata["refined_by"] = "rule"
            cleaned_pairs.append(
                (parent_id, Document(page_content=cleaned, metadata=metadata))
            )
        return self.metadata_enricher.transform_parents(cleaned_pairs, trace=trace)


def run_pipeline(
    rag_system,
    file_path: str,
    *,
    collection: str = "default",
    force: bool = False,
    on_progress: Optional[Callable[[str, int, int], None]] = None,
) -> PipelineResult:
    pipeline = IngestionPipeline(rag_system, collection=collection, force=force)
    trace = IngestionTrace(
        metadata={
            "source_path": file_path,
            "collection": collection,
            "force": force,
        }
    )
    result = pipeline.run(file_path, trace=trace, on_progress=on_progress)
    if not result.skipped:
        trace.finish(success=result.success, error=result.error)
        pipeline.trace_collector.save(trace)
    return result
