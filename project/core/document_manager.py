from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple, Union

import config
from ingestion.document_registry import DocumentRegistry
from ingestion.loaders.document_loader import UniversalDocumentLoader
from ingestion.pipeline import IngestionPipeline, PipelineResult
from ingestion.trace import IngestionTrace, IngestionTraceCollector


@dataclass
class DocumentInfo:
    display_name: str
    stem: str
    source_name: str
    title: Optional[str] = None
    doc_hash: Optional[str] = None
    file_path: Optional[str] = None
    chunk_count: int = 0
    parent_count: int = 0
    processed_at: Optional[str] = None


@dataclass
class DeleteResult:
    success: bool
    display_name: str
    chunks_deleted: int = 0
    parents_deleted: int = 0
    images_deleted: int = 0
    markdown_removed: bool = False
    converted_docx_removed: int = 0
    integrity_removed: bool = False
    registry_removed: bool = False
    errors: List[str] = field(default_factory=list)


SyncAction = Literal["added", "updated", "skipped", "failed"]


@dataclass
class SyncResult:
    action: SyncAction
    stem: str
    display_name: str
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None
    purge: Optional[DeleteResult] = None
    ingest: Optional[PipelineResult] = None
    error: Optional[str] = None


@dataclass
class SyncSummary:
    added: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    results: List[SyncResult] = field(default_factory=list)


class DocumentManager:

    SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".docx", ".doc"}

    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.markdown_dir = Path(config.MARKDOWN_DIR)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = Path(config.INGESTION_IMAGES_DIR)
        self.converted_docx_dir = Path(config.INGESTION_CONVERTED_DOCX_DIR)
        self.pipeline = IngestionPipeline(rag_system)
        self.registry = DocumentRegistry(config.INGESTION_DB_PATH)
        self.trace_collector = IngestionTraceCollector(config.INGESTION_TRACE_FILE)
        self._bootstrap_registry_from_integrity()

    def sync_documents(
        self,
        document_paths: Union[str, List[str]],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> SyncSummary:
        if not document_paths:
            return SyncSummary()

        if isinstance(document_paths, str):
            document_paths = [document_paths]

        document_paths = [
            p for p in document_paths
            if p and Path(p).suffix.lower() in self.SUPPORTED_SUFFIXES
        ]
        if not document_paths:
            return SyncSummary()

        summary = SyncSummary()
        total = len(document_paths)

        for index, doc_path in enumerate(document_paths):
            file_name = Path(doc_path).name
            if progress_callback:
                progress_callback(
                    index / total,
                    f"Syncing {file_name} ({index + 1}/{total})",
                )

            result = self.sync_document(
                doc_path,
                progress_callback=(
                    lambda p, desc, base=index, t=total: progress_callback(
                        (base + p) / t,
                        f"[{base + 1}/{t}] {file_name}: {desc}",
                    )
                    if progress_callback
                    else None
                ),
            )
            summary.results.append(result)

            if result.action == "added":
                summary.added += 1
            elif result.action == "updated":
                summary.updated += 1
            elif result.action == "skipped":
                summary.skipped += 1
            else:
                summary.failed += 1

        if progress_callback:
            progress_callback(1.0, "Sync complete")

        return summary

    def sync_document(
        self,
        file_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> SyncResult:
        path = Path(file_path)
        if path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
            return SyncResult(
                action="failed",
                stem=path.stem,
                display_name=path.name,
                error=f"Unsupported file type: {path.suffix}",
            )

        stem = path.stem
        display_name = path.name

        try:
            new_hash = self.pipeline.integrity_checker.compute_sha256(str(path))
        except (OSError, FileNotFoundError) as exc:
            return SyncResult(
                action="failed",
                stem=stem,
                display_name=display_name,
                error=str(exc),
            )

        record = self.registry.get_by_stem(stem)

        if (
            record
            and record.status == "active"
            and record.content_hash == new_hash
        ):
            return SyncResult(
                action="skipped",
                stem=stem,
                display_name=display_name,
                new_hash=new_hash,
            )

        if record and record.status == "active" and record.content_hash != new_hash:
            purge = self.purge_document(
                stem=stem,
                doc_hash=record.content_hash,
                display_name=record.display_name,
            )
            ingest = self._ingest_file(
                str(path),
                force=True,
                progress_callback=progress_callback,
            )
            if not ingest.success:
                return SyncResult(
                    action="failed",
                    stem=stem,
                    display_name=display_name,
                    old_hash=record.content_hash,
                    new_hash=new_hash,
                    purge=purge,
                    ingest=ingest,
                    error=ingest.error or "Re-ingestion failed after purge",
                )

            self.registry.upsert(
                stem=stem,
                display_name=display_name,
                content_hash=new_hash,
                file_path=str(path),
                child_count=ingest.child_count,
                parent_count=ingest.parent_count,
                status="active",
            )
            return SyncResult(
                action="updated",
                stem=stem,
                display_name=display_name,
                old_hash=record.content_hash,
                new_hash=new_hash,
                purge=purge,
                ingest=ingest,
            )

        ingest = self._ingest_file(
            str(path),
            force=False,
            progress_callback=progress_callback,
        )
        if ingest.skipped:
            self.registry.upsert(
                stem=stem,
                display_name=display_name,
                content_hash=new_hash,
                file_path=str(path),
                status="active",
            )
            return SyncResult(
                action="skipped",
                stem=stem,
                display_name=display_name,
                new_hash=new_hash,
                ingest=ingest,
            )

        if not ingest.success:
            return SyncResult(
                action="failed",
                stem=stem,
                display_name=display_name,
                new_hash=new_hash,
                ingest=ingest,
                error=ingest.error or "Ingestion failed",
            )

        self.registry.upsert(
            stem=stem,
            display_name=display_name,
            content_hash=new_hash,
            file_path=str(path),
            child_count=ingest.child_count,
            parent_count=ingest.parent_count,
            status="active",
        )
        return SyncResult(
            action="added",
            stem=stem,
            display_name=display_name,
            new_hash=new_hash,
            ingest=ingest,
        )

    def add_documents(
        self,
        document_paths,
        progress_callback=None,
        force: bool = False,
    ) -> Tuple[int, int]:
        """Backward-compatible upload API. Returns (added_or_updated, skipped)."""
        if force:
            summary = SyncSummary()
            paths = (
                [document_paths]
                if isinstance(document_paths, str)
                else list(document_paths or [])
            )
            paths = [
                p for p in paths
                if p and Path(p).suffix.lower() in self.SUPPORTED_SUFFIXES
            ]
            for path in paths:
                record = self.registry.get_by_stem(Path(path).stem)
                if record and record.status == "active":
                    self.purge_document(
                        stem=record.stem,
                        doc_hash=record.content_hash,
                        display_name=record.display_name,
                    )
                ingest = self._ingest_file(path, force=True, progress_callback=progress_callback)
                if ingest.success:
                    self.registry.upsert(
                        stem=Path(path).stem,
                        display_name=Path(path).name,
                        content_hash=ingest.doc_id or "",
                        file_path=str(path),
                        child_count=ingest.child_count,
                        parent_count=ingest.parent_count,
                    )
                    summary.added += 1
                else:
                    summary.failed += 1
            return summary.added, summary.skipped

        summary = self.sync_documents(document_paths, progress_callback=progress_callback)
        changed = summary.added + summary.updated
        return changed, summary.skipped

    def purge_document(
        self,
        *,
        stem: str,
        doc_hash: Optional[str] = None,
        display_name: Optional[str] = None,
    ) -> DeleteResult:
        source_name = f"{stem}.pdf"
        result = DeleteResult(
            success=True,
            display_name=display_name or source_name,
        )
        collection_name = self.rag_system.collection_name
        vector_db = self.rag_system.vector_db

        try:
            if doc_hash:
                result.chunks_deleted = vector_db.delete_by_doc_hash(
                    collection_name,
                    doc_hash,
                )
            if result.chunks_deleted == 0:
                result.chunks_deleted = vector_db.delete_by_source(
                    collection_name,
                    source_name,
                )
        except Exception as exc:
            result.errors.append(f"Qdrant 删除失败: {exc}")

        try:
            result.parents_deleted = self.rag_system.parent_store.delete_by_stem(stem)
        except Exception as exc:
            result.errors.append(f"Parent Store 删除失败: {exc}")

        md_path = self.markdown_dir / f"{stem}.md"
        try:
            if md_path.exists():
                md_path.unlink()
                result.markdown_removed = True
        except Exception as exc:
            result.errors.append(f"Markdown 删除失败: {exc}")

        try:
            result.images_deleted = self._delete_image_dirs(doc_hash, stem)
        except Exception as exc:
            result.errors.append(f"图片目录删除失败: {exc}")

        try:
            result.converted_docx_removed = self._delete_converted_docx_dirs(doc_hash)
        except Exception as exc:
            result.errors.append(f"DOCX 转换缓存删除失败: {exc}")

        if doc_hash:
            try:
                result.integrity_removed = self.pipeline.integrity_checker.remove_record(
                    doc_hash
                )
            except Exception as exc:
                result.errors.append(f"Integrity 记录删除失败: {exc}")

        try:
            result.registry_removed = self.registry.remove(stem)
        except Exception as exc:
            result.errors.append(f"Registry 记录删除失败: {exc}")

        if result.errors:
            result.success = False
        return result

    def delete_document(self, display_name: str) -> DeleteResult:
        if not display_name:
            return DeleteResult(success=False, display_name="", errors=["未选择文档"])

        doc_info = self._find_document(display_name)
        if doc_info is None:
            return DeleteResult(
                success=False,
                display_name=display_name,
                errors=[f"未找到文档: {display_name}"],
            )

        return self.purge_document(
            stem=doc_info.stem,
            doc_hash=doc_info.doc_hash,
            display_name=doc_info.display_name,
        )

    def list_documents(self, *, fast: bool = False) -> List[DocumentInfo]:
        docs_by_stem: dict[str, DocumentInfo] = {}

        for record in self.registry.list_active():
            source_name = record.source_name
            chunk_count = self._resolve_child_chunk_count(
                doc_hash=record.content_hash,
                source_name=source_name,
                fast=fast,
            )
            parent_count = len(
                list(Path(config.PARENT_STORE_PATH).glob(f"{record.stem}_parent_*.json"))
            )
            docs_by_stem[record.stem] = DocumentInfo(
                display_name=record.display_name,
                stem=record.stem,
                source_name=source_name,
                doc_hash=record.content_hash,
                file_path=record.file_path,
                chunk_count=chunk_count,
                parent_count=parent_count,
                processed_at=record.processed_at,
            )

        for md_path in sorted(self.markdown_dir.glob("*.md")):
            stem = md_path.stem
            if stem in docs_by_stem:
                continue

            source_name = f"{stem}.pdf"
            parent_count = len(
                list(Path(config.PARENT_STORE_PATH).glob(f"{stem}_parent_*.json"))
            )
            chunk_count = self._resolve_child_chunk_count(
                doc_hash=None,
                source_name=source_name,
                fast=fast,
            )
            docs_by_stem[stem] = DocumentInfo(
                display_name=source_name,
                stem=stem,
                source_name=source_name,
                chunk_count=chunk_count,
                parent_count=parent_count,
            )

        docs = sorted(docs_by_stem.values(), key=lambda item: item.display_name.lower())
        self._enrich_document_titles(docs)
        return docs

    def _ingest_file(
        self,
        file_path: str,
        *,
        force: bool,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> PipelineResult:
        file_name = Path(file_path).name
        self.pipeline.force = force

        trace = IngestionTrace(
            metadata={
                "source_path": file_name,
                "collection": self.pipeline.collection,
                "source": "gradio",
                "force": force,
            }
        )

        def on_progress(stage: str, current: int, total: int) -> None:
            if progress_callback:
                progress_callback(current / total, stage)

        result = self.pipeline.run(file_path, trace=trace, on_progress=on_progress)

        if not result.skipped:
            trace.finish(success=result.success, error=result.error)
            self.trace_collector.save(trace)
        elif result.success:
            trace.finish(success=True)
            self.trace_collector.save(trace)

        if not result.success and not result.skipped:
            print(f"Error processing {file_path}: {result.error}")

        return result

    def _bootstrap_registry_from_integrity(self) -> None:
        if self.registry.list_active():
            return

        records = self.pipeline.integrity_checker.list_processed()
        if not records:
            return

        latest_by_stem: dict[str, dict] = {}
        for record in records:
            file_path = record.get("file_path", "")
            if not file_path:
                continue
            stem = Path(file_path).stem
            existing = latest_by_stem.get(stem)
            if existing is None or record.get("updated_at", "") >= existing.get(
                "updated_at", ""
            ):
                latest_by_stem[stem] = record

        for stem, record in latest_by_stem.items():
            file_path = record.get("file_path", "")
            self.registry.upsert(
                stem=stem,
                display_name=Path(file_path).name if file_path else f"{stem}.pdf",
                content_hash=record["file_hash"],
                file_path=file_path,
                status="active",
            )

    def _read_document_title(self, stem: str) -> Optional[str]:
        md_path = self.markdown_dir / f"{stem}.md"
        if not md_path.exists():
            return None
        try:
            return UniversalDocumentLoader._extract_title(
                md_path.read_text(encoding="utf-8")
            )
        except OSError:
            return None

    def _enrich_document_titles(self, docs: List[DocumentInfo]) -> None:
        for doc in docs:
            if not doc.title:
                doc.title = self._read_document_title(doc.stem)

    def document_choices(self, *, fast: bool = False) -> List[str]:
        return [doc.display_name for doc in self.list_documents(fast=fast)]

    def _resolve_child_chunk_count(
        self,
        *,
        doc_hash: Optional[str],
        source_name: str,
        fast: bool,
    ) -> int:
        if not fast:
            if doc_hash:
                count = self.rag_system.vector_db.count_chunks_by_doc_hash(
                    self.rag_system.collection_name,
                    doc_hash,
                )
                if count > 0:
                    return count
            return self.rag_system.vector_db.count_chunks_by_source(
                self.rag_system.collection_name,
                source_name,
            )

        return self._child_count_from_traces(source_name)

    @staticmethod
    def _child_count_from_traces(source_name: str) -> int:
        import json

        trace_file = Path(config.INGESTION_TRACE_FILE)
        if not trace_file.is_file():
            return 0

        latest_count = 0
        tail_bytes = min(trace_file.stat().st_size, 512_000)
        with trace_file.open("rb") as handle:
            handle.seek(-tail_bytes, 2)
            chunk = handle.read().decode("utf-8", errors="ignore")
        lines = chunk.splitlines()
        if tail_bytes < trace_file.stat().st_size and lines:
            lines = lines[1:]

        for line in lines:
            if not line.strip():
                continue
            try:
                trace = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not trace.get("success"):
                continue
            source_path = trace.get("metadata", {}).get("source_path", "")
            if Path(source_path).name != source_name:
                continue
            upsert = next(
                (stage for stage in trace.get("stages", []) if stage.get("stage") == "upsert"),
                None,
            )
            if upsert is None:
                continue
            child_count = upsert.get("data", {}).get("child_count")
            if child_count is not None:
                latest_count = int(child_count)
        return latest_count

    def _find_document(self, display_name: str) -> Optional[DocumentInfo]:
        for doc in self.list_documents():
            if doc.display_name == display_name:
                return doc
        return None

    def _delete_image_dirs(self, doc_hash: Optional[str], stem: str) -> int:
        deleted_dirs = 0
        candidates = []
        if doc_hash:
            candidates.append(self.images_dir / doc_hash[:16])
            candidates.append(self.images_dir / doc_hash)

        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                shutil.rmtree(candidate, ignore_errors=True)
                deleted_dirs += 1

        if deleted_dirs == 0 and self.images_dir.exists():
            for image_dir in self.images_dir.iterdir():
                if not image_dir.is_dir():
                    continue
                if any(image_dir.glob(f"{stem}*")) or any(image_dir.glob(f"*_{stem}_*")):
                    shutil.rmtree(image_dir, ignore_errors=True)
                    deleted_dirs += 1
        return deleted_dirs

    def _delete_converted_docx_dirs(self, doc_hash: Optional[str]) -> int:
        if not doc_hash:
            return 0
        deleted = 0
        for candidate in (
            self.converted_docx_dir / doc_hash[:16],
            self.converted_docx_dir / doc_hash,
        ):
            if candidate.exists() and candidate.is_dir():
                shutil.rmtree(candidate, ignore_errors=True)
                deleted += 1
        return deleted

    def get_markdown_files(self):
        if not self.markdown_dir.exists():
            return []
        return sorted([p.name.replace(".md", ".pdf") for p in self.markdown_dir.glob("*.md")])

    def clear_all(self):
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        from utils import clear_directory_contents

        clear_directory_contents(self.markdown_dir)

        self.rag_system.parent_store.clear_store()
        self.rag_system.vector_db.delete_collection(self.rag_system.collection_name)
        self.rag_system.vector_db.create_collection(self.rag_system.collection_name)

        if self.images_dir.exists():
            clear_directory_contents(self.images_dir)

        if self.converted_docx_dir.exists():
            clear_directory_contents(self.converted_docx_dir)

        for record in self.pipeline.integrity_checker.list_processed():
            self.pipeline.integrity_checker.remove_record(record["file_hash"])

        self.registry.clear()

    def list_ingestion_traces(self, limit: int = 20):
        trace_file = Path(config.INGESTION_TRACE_FILE)
        if not trace_file.exists():
            return []
        lines = trace_file.read_text(encoding="utf-8").splitlines()
        traces = []
        import json

        for line in reversed(lines[-limit:]):
            if line.strip():
                traces.append(json.loads(line))
        return traces

    def format_document_table(self, *, fast: bool = False) -> str:
        docs = self.list_documents(fast=fast)
        if not docs:
            return "📭 知识库中暂无文档"
        lines = ["| 文档 | 标题 | 子块 | 父块 | 状态 |", "| --- | --- | ---: | ---: | --- |"]
        for doc in docs:
            status = "已索引" if doc.doc_hash else "仅 Markdown"
            title = doc.title or "—"
            lines.append(
                f"| {doc.display_name} | {title} | {doc.chunk_count} | {doc.parent_count} | {status} |"
            )
        return "\n".join(lines)
