"""Ingestion trace context for stage-level observability."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


@dataclass
class IngestionTrace:
    trace_type: Literal["ingestion"] = "ingestion"
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: Optional[str] = None
    stages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: Optional[bool] = None
    error: Optional[str] = None
    _start_mono: float = field(default_factory=time.monotonic, repr=False)
    _finish_mono: Optional[float] = field(default=None, repr=False)

    def record_stage(
        self,
        stage_name: str,
        data: Dict[str, Any],
        elapsed_ms: Optional[float] = None,
    ) -> None:
        entry: Dict[str, Any] = {
            "stage": stage_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        if elapsed_ms is not None:
            entry["elapsed_ms"] = round(elapsed_ms, 2)
        self.stages.append(entry)

    def finish(self, *, success: bool, error: Optional[str] = None) -> None:
        self._finish_mono = time.monotonic()
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.success = success
        self.error = error

    def elapsed_ms(self) -> float:
        end = self._finish_mono if self._finish_mono is not None else time.monotonic()
        return (end - self._start_mono) * 1000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_ms": round(self.elapsed_ms(), 2),
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
            "stages": self.stages,
        }


class IngestionTraceCollector:
    """Append ingestion traces to a JSON Lines file."""

    def __init__(self, trace_file: str):
        self.trace_file = Path(trace_file)
        self.trace_file.parent.mkdir(parents=True, exist_ok=True)

    def save(self, trace: IngestionTrace) -> None:
        with self.trace_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
