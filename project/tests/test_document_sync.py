"""Tests for document-level incremental sync (registry + purge + sync)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.document_manager import DocumentManager
from ingestion.document_registry import DocumentRegistry
from ingestion.pipeline import PipelineResult


def test_document_registry_upsert_and_get():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        registry = DocumentRegistry(str(db_path))
        registry.upsert(
            stem="report",
            display_name="report.pdf",
            content_hash="hash_a",
            file_path="/tmp/report.pdf",
            child_count=10,
            parent_count=2,
        )
        record = registry.get_by_stem("report")
        assert record is not None
        assert record.content_hash == "hash_a"
        assert record.display_name == "report.pdf"
        assert len(registry.list_active()) == 1

        registry.upsert(
            stem="report",
            display_name="report.pdf",
            content_hash="hash_b",
            child_count=12,
            parent_count=3,
        )
        updated = registry.get_by_stem("report")
        assert updated.content_hash == "hash_b"
        assert updated.child_count == 12

        assert registry.remove("report")
        assert registry.get_by_stem("report") is None


def test_purge_document_clears_storage_layers():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        markdown_dir = tmp_path / "markdown_docs"
        parent_store = tmp_path / "parent_store"
        images_dir = tmp_path / "images"
        converted_dir = tmp_path / "converted_docx"
        db_path = tmp_path / "ingestion.db"

        markdown_dir.mkdir()
        parent_store.mkdir()
        images_dir.mkdir()
        converted_dir.mkdir()

        stem = "demo"
        doc_hash = "a" * 64
        (markdown_dir / f"{stem}.md").write_text("# Demo", encoding="utf-8")
        (parent_store / f"{stem}_parent_0.json").write_text("{}", encoding="utf-8")
        (images_dir / doc_hash[:16]).mkdir()
        (converted_dir / doc_hash[:16]).mkdir()

        registry = DocumentRegistry(str(db_path))
        registry.upsert(
            stem=stem,
            display_name="demo.pdf",
            content_hash=doc_hash,
        )

        rag_system = MagicMock()
        rag_system.collection_name = "document_child_chunks"
        rag_system.vector_db.delete_by_doc_hash.return_value = 3
        rag_system.parent_store.delete_by_stem.return_value = 1

        with patch("core.document_manager.config") as mock_config:
            mock_config.MARKDOWN_DIR = str(markdown_dir)
            mock_config.PARENT_STORE_PATH = str(parent_store)
            mock_config.INGESTION_IMAGES_DIR = str(images_dir)
            mock_config.INGESTION_CONVERTED_DOCX_DIR = str(converted_dir)
            mock_config.INGESTION_DB_PATH = str(db_path)
            mock_config.INGESTION_TRACE_FILE = str(tmp_path / "traces.jsonl")

            manager = DocumentManager.__new__(DocumentManager)
            manager.rag_system = rag_system
            manager.markdown_dir = markdown_dir
            manager.images_dir = images_dir
            manager.converted_docx_dir = converted_dir
            manager.registry = registry
            manager.pipeline = MagicMock()
            manager.pipeline.integrity_checker.remove_record.return_value = True
            manager.trace_collector = MagicMock()

            result = manager.purge_document(
                stem=stem,
                doc_hash=doc_hash,
                display_name="demo.pdf",
            )

        assert result.success
        assert result.chunks_deleted == 3
        assert result.parents_deleted == 1
        assert result.markdown_removed
        assert result.images_deleted >= 1
        assert result.converted_docx_removed >= 1
        assert result.registry_removed
        assert not (markdown_dir / f"{stem}.md").exists()
        assert registry.get_by_stem(stem) is None


def test_sync_document_skips_unchanged_hash():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        file_path = tmp_path / "report.pdf"
        file_path.write_bytes(b"same content")
        db_path = tmp_path / "ingestion.db"

        from ingestion.file_integrity import FileIntegrityChecker

        checker = FileIntegrityChecker(str(db_path))
        content_hash = checker.compute_sha256(str(file_path))

        registry = DocumentRegistry(str(db_path))
        registry.upsert(
            stem="report",
            display_name="report.pdf",
            content_hash=content_hash,
        )

        rag_system = MagicMock()
        with patch("core.document_manager.config") as mock_config, patch.object(
            DocumentManager, "_bootstrap_registry_from_integrity", return_value=None
        ):
            mock_config.MARKDOWN_DIR = str(tmp_path / "md")
            mock_config.INGESTION_IMAGES_DIR = str(tmp_path / "images")
            mock_config.INGESTION_CONVERTED_DOCX_DIR = str(tmp_path / "converted")
            mock_config.INGESTION_DB_PATH = str(db_path)
            mock_config.INGESTION_TRACE_FILE = str(tmp_path / "traces.jsonl")
            mock_config.PARENT_STORE_PATH = str(tmp_path / "parents")

            manager = DocumentManager(rag_system)
            manager.registry = registry
            manager.pipeline = MagicMock()
            manager.pipeline.integrity_checker = checker

            result = manager.sync_document(str(file_path))

        assert result.action == "skipped"
        manager.pipeline.run.assert_not_called()


def test_sync_document_updates_when_hash_changes():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        file_path = tmp_path / "report.pdf"
        file_path.write_bytes(b"version 1")
        db_path = tmp_path / "ingestion.db"

        from ingestion.file_integrity import FileIntegrityChecker

        checker = FileIntegrityChecker(str(db_path))
        old_hash = checker.compute_sha256(str(file_path))

        registry = DocumentRegistry(str(db_path))
        registry.upsert(
            stem="report",
            display_name="report.pdf",
            content_hash=old_hash,
        )

        file_path.write_bytes(b"version 2")
        new_hash = checker.compute_sha256(str(file_path))

        rag_system = MagicMock()
        rag_system.collection_name = "document_child_chunks"
        rag_system.vector_db.delete_by_doc_hash.return_value = 5
        rag_system.parent_store.delete_by_stem.return_value = 2

        with patch("core.document_manager.config") as mock_config, patch.object(
            DocumentManager, "_bootstrap_registry_from_integrity", return_value=None
        ):
            mock_config.MARKDOWN_DIR = str(tmp_path / "md")
            Path(mock_config.MARKDOWN_DIR).mkdir(parents=True)
            mock_config.PARENT_STORE_PATH = str(tmp_path / "parents")
            mock_config.INGESTION_IMAGES_DIR = str(tmp_path / "images")
            mock_config.INGESTION_CONVERTED_DOCX_DIR = str(tmp_path / "converted")
            mock_config.INGESTION_DB_PATH = str(db_path)
            mock_config.INGESTION_TRACE_FILE = str(tmp_path / "traces.jsonl")

            manager = DocumentManager(rag_system)
            manager.registry = registry
            manager.pipeline = MagicMock()
            manager.pipeline.integrity_checker = MagicMock()
            manager.pipeline.integrity_checker.compute_sha256.return_value = new_hash
            manager.trace_collector = MagicMock()
            manager.pipeline.integrity_checker.remove_record.return_value = True
            manager.pipeline.run.return_value = PipelineResult(
                success=True,
                file_path=str(file_path),
                doc_id=new_hash,
                parent_count=2,
                child_count=8,
            )

            with patch.object(manager, "purge_document") as purge_spy:
                purge_spy.return_value = MagicMock(success=True)
                result = manager.sync_document(str(file_path))

        assert result.action == "updated"
        assert result.old_hash == old_hash
        assert result.new_hash == new_hash
        purge_spy.assert_called_once_with(
            stem="report",
            doc_hash=old_hash,
            display_name="report.pdf",
        )
        manager.pipeline.run.assert_called_once()
        record = registry.get_by_stem("report")
        assert record is not None
        assert record.content_hash == new_hash


if __name__ == "__main__":
    test_document_registry_upsert_and_get()
    test_purge_document_clears_storage_layers()
    test_sync_document_skips_unchanged_hash()
    test_sync_document_updates_when_hash_changes()
    print("All document sync tests passed.")
