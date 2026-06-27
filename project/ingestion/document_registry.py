"""Logical document registry: stem -> current content_hash for incremental sync."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


@dataclass
class DocumentRecord:
    stem: str
    display_name: str
    content_hash: str
    file_path: Optional[str] = None
    status: str = "active"
    child_count: int = 0
    parent_count: int = 0
    processed_at: Optional[str] = None
    updated_at: Optional[str] = None

    @property
    def source_name(self) -> str:
        return f"{self.stem}.pdf"


class DocumentRegistry:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_database()

    def _ensure_database(self) -> None:
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS document_registry (
                    stem TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    file_path TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    child_count INTEGER DEFAULT 0,
                    parent_count INTEGER DEFAULT 0,
                    processed_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_registry_hash ON document_registry(content_hash)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_registry_status ON document_registry(status)"
            )
            conn.commit()
        finally:
            conn.close()

    def get_by_stem(self, stem: str) -> Optional[DocumentRecord]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                """
                SELECT stem, display_name, content_hash, file_path, status,
                       child_count, parent_count, processed_at, updated_at
                FROM document_registry
                WHERE stem = ?
                """,
                (stem,),
            )
            row = cursor.fetchone()
            return self._row_to_record(row) if row else None
        finally:
            conn.close()

    def list_active(self) -> List[DocumentRecord]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                """
                SELECT stem, display_name, content_hash, file_path, status,
                       child_count, parent_count, processed_at, updated_at
                FROM document_registry
                WHERE status = 'active'
                ORDER BY display_name COLLATE NOCASE ASC
                """
            )
            return [self._row_to_record(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def upsert(
        self,
        stem: str,
        display_name: str,
        content_hash: str,
        *,
        file_path: Optional[str] = None,
        child_count: int = 0,
        parent_count: int = 0,
        status: str = "active",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT processed_at FROM document_registry WHERE stem = ?",
                (stem,),
            )
            row = cursor.fetchone()
            processed_at = row[0] if row else now
            conn.execute(
                """
                INSERT INTO document_registry (
                    stem, display_name, content_hash, file_path, status,
                    child_count, parent_count, processed_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(stem) DO UPDATE SET
                    display_name = excluded.display_name,
                    content_hash = excluded.content_hash,
                    file_path = excluded.file_path,
                    status = excluded.status,
                    child_count = excluded.child_count,
                    parent_count = excluded.parent_count,
                    updated_at = excluded.updated_at
                """,
                (
                    stem,
                    display_name,
                    content_hash,
                    file_path,
                    status,
                    child_count,
                    parent_count,
                    processed_at,
                    now,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def mark_deleted(self, stem: str) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """
                UPDATE document_registry
                SET status = 'deleted', updated_at = ?
                WHERE stem = ?
                """,
                (now, stem),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def remove(self, stem: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "DELETE FROM document_registry WHERE stem = ?",
                (stem,),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def clear(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM document_registry")
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            stem=row["stem"],
            display_name=row["display_name"],
            content_hash=row["content_hash"],
            file_path=row["file_path"],
            status=row["status"],
            child_count=int(row["child_count"] or 0),
            parent_count=int(row["parent_count"] or 0),
            processed_at=row["processed_at"],
            updated_at=row["updated_at"],
        )
