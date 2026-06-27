"""Pluggable rerankers for second-stage retrieval precision.

Supports Cross-Encoder and LLM-based reranking, following the design of
MODULAR-RAG-MCP-SERVER with a lighter integration for agentic-rag.
"""

from __future__ import annotations

import json
import logging
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import config

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "rerank.txt"
_CROSS_ENCODER_LOCK = threading.Lock()
_CROSS_ENCODER_MODEL: Any = None
_CROSS_ENCODER_MODEL_NAME: Optional[str] = None


class RerankError(RuntimeError):
    """Raised when reranking fails."""


class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Return candidates sorted by relevance (highest first), at most top_k."""


class NoneReranker(BaseReranker):
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        return list(candidates[:top_k])


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or config.RERANK_MODEL
        self.model = self._load_model(self.model_name)

    @staticmethod
    def _load_model(model_name: str) -> Any:
        global _CROSS_ENCODER_MODEL, _CROSS_ENCODER_MODEL_NAME
        with _CROSS_ENCODER_LOCK:
            if _CROSS_ENCODER_MODEL is not None and _CROSS_ENCODER_MODEL_NAME == model_name:
                return _CROSS_ENCODER_MODEL
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for Cross-Encoder reranking."
                ) from exc
            logger.info("Loading Cross-Encoder rerank model: %s", model_name)
            _CROSS_ENCODER_MODEL = CrossEncoder(model_name)
            _CROSS_ENCODER_MODEL_NAME = model_name
            return _CROSS_ENCODER_MODEL

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        if len(candidates) == 1:
            only = dict(candidates[0])
            only.setdefault("rerank_score", 1.0)
            return [only]

        pairs = [(query, str(c.get("text") or c.get("content") or "")) for c in candidates]
        scores = self.model.predict(pairs)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()

        scored: List[Dict[str, Any]] = []
        for candidate, score in zip(candidates, scores):
            item = dict(candidate)
            item["rerank_score"] = float(score)
            scored.append(item)
        scored.sort(key=lambda item: item["rerank_score"], reverse=True)
        return scored[:top_k]


class LLMReranker(BaseReranker):
    def __init__(self, prompt_path: Optional[Path] = None) -> None:
        self.prompt_template = (prompt_path or _PROMPT_PATH).read_text(encoding="utf-8")

    def _build_prompt(self, query: str, candidates: List[Dict[str, Any]]) -> str:
        passages = []
        for index, candidate in enumerate(candidates):
            passage_id = candidate.get("id", f"passage_{index}")
            text = candidate.get("text") or candidate.get("content") or ""
            passages.append(f"Passage ID: {passage_id}\nText: {text}\n")
        return (
            f"{self.prompt_template}\n\n"
            f"Query: {query}\n\n"
            f"Passages:\n{''.join(passages)}\n"
            "Output your response as a JSON array of objects, one per passage."
        )

    @staticmethod
    def _parse_response(response_text: str) -> List[Dict[str, Any]]:
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise RerankError(f"Expected JSON array, got {type(parsed).__name__}")
        for index, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise RerankError(f"Item {index} is not an object")
            if "passage_id" not in item or "score" not in item:
                raise RerankError(f"Item {index} missing passage_id or score")
        return parsed

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        if len(candidates) == 1:
            only = dict(candidates[0])
            only.setdefault("rerank_score", 1.0)
            return [only]

        from core.rag_system import create_llm

        llm = create_llm()
        prompt = self._build_prompt(query, candidates)
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        parsed = self._parse_response(str(content))

        id_to_candidate = {
            str(candidate.get("id", f"passage_{index}")): candidate
            for index, candidate in enumerate(candidates)
        }
        reranked: List[Dict[str, Any]] = []
        for item in parsed:
            passage_id = str(item["passage_id"])
            if passage_id not in id_to_candidate:
                continue
            candidate = dict(id_to_candidate[passage_id])
            candidate["rerank_score"] = float(item["score"])
            reranked.append(candidate)
        reranked.sort(key=lambda item: item.get("rerank_score", 0.0), reverse=True)
        return reranked[:top_k]


def create_reranker() -> BaseReranker:
    if not config.RERANK_ENABLED or config.RERANK_PROVIDER in {"none", "disabled"}:
        return NoneReranker()
    if config.RERANK_PROVIDER == "cross_encoder":
        return CrossEncoderReranker()
    if config.RERANK_PROVIDER == "llm":
        return LLMReranker()
    raise ValueError(
        f"Unsupported RERANK_PROVIDER '{config.RERANK_PROVIDER}'. "
        "Use: none, cross_encoder, llm."
    )


_RERANKER_INSTANCE: Optional[BaseReranker] = None
_RERANKER_LOCK = threading.Lock()


def get_reranker() -> BaseReranker:
    global _RERANKER_INSTANCE
    with _RERANKER_LOCK:
        if _RERANKER_INSTANCE is None:
            _RERANKER_INSTANCE = create_reranker()
        return _RERANKER_INSTANCE


def reset_reranker_cache() -> None:
    """Clear cached reranker (useful after config changes in tests)."""
    global _RERANKER_INSTANCE, _CROSS_ENCODER_MODEL, _CROSS_ENCODER_MODEL_NAME
    with _RERANKER_LOCK:
        _RERANKER_INSTANCE = None
    with _CROSS_ENCODER_LOCK:
        _CROSS_ENCODER_MODEL = None
        _CROSS_ENCODER_MODEL_NAME = None
