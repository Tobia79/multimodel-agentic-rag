"""SHA256-based file integrity tracking for incremental ingestion."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class FileIntegrityChecker:
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
                CREATE TABLE IF NOT EXISTS ingestion_history (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    collection TEXT,
                    error_msg TEXT,
                    processed_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_status ON ingestion_history(status)"
            )
            conn.commit()
        finally:
            conn.close()

    def compute_sha256(self, file_path: str) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise IOError(f"Path is not a file: {file_path}")

        digest = hashlib.sha256()
        with open(file_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def should_skip(self, file_hash: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT status FROM ingestion_history WHERE file_hash = ?",
                (file_hash,),
            )
            row = cursor.fetchone()
            return row is not None and row[0] == "success"
        finally:
            conn.close()

    def mark_success(
        self,
        file_hash: str,
        file_path: str,
        collection: Optional[str] = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT processed_at FROM ingestion_history WHERE file_hash = ?",
                (file_hash,),
            )
            row = cursor.fetchone()
            if row:
                conn.execute(
                    """
                    UPDATE ingestion_history
                    SET file_path = ?, status = 'success', collection = ?,
                        error_msg = NULL, updated_at = ?
                    WHERE file_hash = ?
                    """,
                    (file_path, collection, now, file_hash),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO ingestion_history
                    (file_hash, file_path, status, collection, error_msg, processed_at, updated_at)
                    VALUES (?, ?, 'success', ?, NULL, ?, ?)
                    """,
                    (file_hash, file_path, collection, now, now),
                )
            conn.commit()
        finally:
            conn.close()

    def mark_failed(self, file_hash: str, file_path: str, error_msg: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT processed_at FROM ingestion_history WHERE file_hash = ?",
                (file_hash,),
            )
            row = cursor.fetchone()
            if row:
                conn.execute(
                    """
                    UPDATE ingestion_history
                    SET file_path = ?, status = 'failed', error_msg = ?, updated_at = ?
                    WHERE file_hash = ?
                    """,
                    (file_path, error_msg, now, file_hash),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO ingestion_history
                    (file_hash, file_path, status, collection, error_msg, processed_at, updated_at)
                    VALUES (?, ?, 'failed', NULL, ?, ?, ?)
                    """,
                    (file_hash, file_path, error_msg, now, now),
                )
            conn.commit()
        finally:
            conn.close()

    def remove_record(self, file_hash: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "DELETE FROM ingestion_history WHERE file_hash = ?",
                (file_hash,),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def list_processed(self, collection: Optional[str] = None) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            query = (
                "SELECT file_hash, file_path, collection, processed_at, updated_at "
                "FROM ingestion_history WHERE status = 'success'"
            )
            params: List[str] = []
            if collection is not None:
                query += " AND collection = ?"
                params.append(collection)
            query += " ORDER BY processed_at ASC"
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
