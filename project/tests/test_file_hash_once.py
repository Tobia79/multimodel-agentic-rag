"""Verify file_hash is computed once in pipeline integrity and reused in load."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from ingestion.file_integrity import FileIntegrityChecker
from ingestion.loaders.document_loader import UniversalDocumentLoader
from ingestion.pipeline import IngestionPipeline


def _sample_md() -> Path:
    sample = PROJECT_DIR.parent / "markdown_docs" / "blockchain.md"
    if sample.exists():
        tmp = Path(tempfile.mkdtemp()) / "blockchain_upload.md"
        tmp.write_bytes(sample.read_bytes())
        return tmp
    tmp = Path(tempfile.mkdtemp()) / "sample.md"
    tmp.write_text("# Test\n\nHello world.\n", encoding="utf-8")
    return tmp


def test_loader_skips_hash_when_provided() -> None:
    md_path = _sample_md()
    checker = FileIntegrityChecker(str(Path(tempfile.mkdtemp()) / "test.db"))
    expected = checker.compute_sha256(str(md_path))

    call_count = {"n": 0}
    original = UniversalDocumentLoader.compute_file_hash

    def counting_compute(path):
        call_count["n"] += 1
        return original(path)

    loader = UniversalDocumentLoader()
    with patch.object(UniversalDocumentLoader, "compute_file_hash", side_effect=counting_compute):
        doc = loader.copy_markdown_file(md_path, file_hash=expected)

    assert call_count["n"] == 0, (
        f"compute_file_hash should not run when file_hash is provided, got {call_count['n']} calls"
    )
    assert doc.file_hash == expected
    assert doc.metadata["doc_hash"] == expected
    print("PASS test_loader_skips_hash_when_provided")


def test_loader_computes_hash_when_not_provided() -> None:
    md_path = _sample_md()
    call_count = {"n": 0}
    original = UniversalDocumentLoader.compute_file_hash

    def counting_compute(path):
        call_count["n"] += 1
        return original(path)

    loader = UniversalDocumentLoader()
    with patch.object(UniversalDocumentLoader, "compute_file_hash", side_effect=counting_compute):
        doc = loader.load(md_path)

    assert call_count["n"] == 1, f"expected 1 hash computation, got {call_count['n']}"
    assert doc.file_hash
    print("PASS test_loader_computes_hash_when_not_provided")


def test_pipeline_load_reuses_integrity_hash() -> None:
    md_path = _sample_md()
    checker = FileIntegrityChecker(str(Path(tempfile.mkdtemp()) / "test.db"))
    integrity_hash = checker.compute_sha256(str(md_path))

    pipeline = IngestionPipeline(MagicMock())

    def fail_if_compute(path):
        raise AssertionError(
            f"compute_file_hash must not be called during pipeline load; path={path}"
        )

    with patch.object(UniversalDocumentLoader, "compute_file_hash", side_effect=fail_if_compute):
        loaded = pipeline._load_document(md_path, file_hash=integrity_hash)

    assert loaded.file_hash == integrity_hash
    assert loaded.metadata["doc_hash"] == integrity_hash
    print("PASS test_pipeline_load_reuses_integrity_hash")


def test_integrity_and_load_single_hash_via_pipeline_run() -> None:
    """Count SHA256 reads: integrity once, load zero extra."""
    md_path = _sample_md()
    db_path = Path(tempfile.mkdtemp()) / "ingestion_test.db"

    sha_calls = {"n": 0}
    original_sha = FileIntegrityChecker.compute_sha256

    def counting_sha(self, file_path: str) -> str:
        sha_calls["n"] += 1
        return original_sha(self, file_path)

    rag = MagicMock()
    rag.chunker = MagicMock()
    rag.chunker.create_chunks_single.return_value = ([], [])
    pipeline = IngestionPipeline(rag)
    pipeline.force = True

    with patch.object(FileIntegrityChecker, "compute_sha256", counting_sha):
        with patch.object(UniversalDocumentLoader, "compute_file_hash") as loader_hash:
            loader_hash.side_effect = AssertionError("loader should not recompute hash")
            try:
                pipeline.run(str(md_path))
            except ValueError:
                pass  # expected: no child chunks from empty mock

    assert sha_calls["n"] == 1, f"integrity should compute SHA256 once, got {sha_calls['n']}"
    loader_hash.assert_not_called()
    print("PASS test_integrity_and_load_single_hash_via_pipeline_run")


if __name__ == "__main__":
    test_loader_skips_hash_when_provided()
    test_loader_computes_hash_when_not_provided()
    test_pipeline_load_reuses_integrity_hash()
    test_integrity_and_load_single_hash_via_pipeline_run()
    print("\nAll file_hash reuse tests passed.")
