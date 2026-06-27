"""RAGAS + custom retrieval evaluation for the Agentic RAG system.

Port of ``notebooks/evaluation.ipynb`` into a reusable module. Typical flow:

1. Load curated QA records from ``notebooks/data/curated_ragas_qa.json``.
2. Run each question through the agentic RAG pipeline.
3. Score retrieval with custom metrics (hit_rate, mrr) and outputs with five RAGAS metrics.
4. Save ``ragas_evaluation_dataset.csv`` and ``rag_evaluation_results.csv``.

CLI::

    python -m core.evaluation
    python -m core.evaluation --sample 5 --skip-query --dataset path/to/saved.csv
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import logging
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from openai import AsyncOpenAI
from tqdm import tqdm

import config
from core.custom_evaluator import (
    CUSTOM_METRIC_NAMES,
    evaluate_custom_metrics,
    extract_ids_from_documents,
)
from core.rag_system import RAGSystem
from core.retrieval import retrieve_child_documents

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(config.MARKDOWN_DIR).parent
_PROJECT_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_QA_PATH = _REPO_ROOT / "notebooks" / "data" / "curated_ragas_qa.json"
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "data" / "evaluation"
_OUTPUT_DIR_DISPLAY = "data/evaluation"
_DATASET_CSV_CANDIDATES = (
    _DEFAULT_OUTPUT_DIR / "ragas_evaluation_dataset.csv",
    _REPO_ROOT / "notebooks" / "ragas_evaluation_dataset.csv",
)
_REQUIRED_MARKDOWN = frozenset({"javascript_tutorial.md", "blockchain.md", "fortinet.md"})

RAGAS_METRIC_NAMES = (
    "answer_accuracy",
    "context_relevance",
    "response_groundedness",
    "context_precision",
    "context_recall",
)
METRIC_NAMES = RAGAS_METRIC_NAMES

ALL_METRIC_NAMES = CUSTOM_METRIC_NAMES + RAGAS_METRIC_NAMES

METRIC_LABELS_ZH = {
    "hit_rate": "命中率 (Hit Rate)",
    "mrr": "MRR",
    "answer_accuracy": "答案准确度",
    "context_relevance": "上下文相关性",
    "response_groundedness": "回答 grounded 度",
    "context_precision": "上下文精确度",
    "context_recall": "上下文召回率",
}

METRIC_LABELS_EN = {
    "hit_rate": "Hit\nRate",
    "mrr": "MRR",
    "answer_accuracy": "Answer\nAccuracy",
    "context_relevance": "Context\nRelevance",
    "response_groundedness": "Response\nGroundedness",
    "context_precision": "Context\nPrecision",
    "context_recall": "Context\nRecall",
}

EvaluationEvent = Union["EvaluationProgress", "MetricScoreProgress", "CustomMetricProgress"]
ProgressCallback = Callable[[EvaluationEvent], None]


def ensure_evaluation_output_dir() -> Path:
    _DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _DEFAULT_OUTPUT_DIR


def display_repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(_REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def format_retrieval_config_line() -> str:
    return (
        f"- **检索**：Dense `{config.DENSE_TOP_K}` + Sparse `{config.SPARSE_TOP_K}` "
        f"→ RRF(k=`{config.RRF_K}`) → fusion `{config.FUSION_TOP_K}`"
    )


def format_confidence_config_line() -> str:
    if not config.CONFIDENCE_ENABLED:
        return "- **置信度路由**：关闭"
    secondary = "开启" if config.CONFIDENCE_SECONDARY_RETRIEVAL else "关闭"
    if config.CONFIDENCE_SECONDARY_RETRIEVAL:
        secondary_detail = (
            f"（top<{config.CONFIDENCE_RERANK_LOW_THRESHOLD:g} 或空结果；"
            f"Dense/Sparse→{config.CONFIDENCE_SECONDARY_DENSE_TOP_K}/"
            f"{config.CONFIDENCE_SECONDARY_SPARSE_TOP_K}）"
        )
    else:
        secondary_detail = ""
    if config.CONFIDENCE_LLM_ENABLED:
        llm_part = (
            f"粗估细估灰区 "
            f"{config.CONFIDENCE_RERANK_GRAY_LOW:g}–{config.CONFIDENCE_RERANK_GRAY_HIGH:g}"
        )
    else:
        llm_part = "仅 Rerank（无 LLM 细估）"
    retry_part = (
        "；中档 Agent 再检索=开启"
        if config.CONFIDENCE_AGENT_RETRY_ON_MEDIUM
        else "；中档 Agent 再检索=关闭"
    )
    web_part = (
        "；低置信 Web 搜索=开启"
        if config.WEB_SEARCH_ENABLED and config.CONFIDENCE_WEB_SEARCH_ON_LOW
        else "；低置信 Web 搜索=关闭"
    )
    return (
        f"- **置信度路由**：扩池二次检索={secondary}{secondary_detail}；"
        f"{llm_part}；"
        f"细估分档：高≥{config.CONFIDENCE_HIGH_THRESHOLD:g}，"
        f"中≥{config.CONFIDENCE_LOW_THRESHOLD:g}且<{config.CONFIDENCE_HIGH_THRESHOLD:g}（再检索），"
        f"低<{config.CONFIDENCE_LOW_THRESHOLD:g}{retry_part}{web_part}"
    )


def format_rerank_config_line() -> str:
    if not config.RERANK_ENABLED or config.RERANK_PROVIDER in {"none", "disabled"}:
        return "- **Rerank**：关闭"
    if config.RERANK_PROVIDER == "cross_encoder":
        return (
            f"- **Rerank**：`cross_encoder` / `{config.RERANK_MODEL}` "
            f"(候选 ×{config.RERANK_CANDIDATE_MULTIPLIER})"
        )
    if config.RERANK_PROVIDER == "llm":
        return f"- **Rerank**：`llm` / `{config.LLM_MODEL}`"
    return f"- **Rerank**：`{config.RERANK_PROVIDER}`"


def format_output_dir_line() -> str:
    ensure_evaluation_output_dir()
    return f"- **输出目录**：`{_OUTPUT_DIR_DISPLAY}/`"


def find_dataset_csv(custom_path: Optional[Path] = None) -> Optional[Path]:
    if custom_path is not None:
        path = Path(custom_path)
        return path if path.is_file() else None
    for candidate in _DATASET_CSV_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def format_dataset_status(custom_path: Optional[Path] = None) -> str:
    resolved = find_dataset_csv(custom_path)
    if resolved:
        return f"`{display_repo_path(resolved)}`"
    searched = "、".join(f"`{display_repo_path(path)}`" for path in _DATASET_CSV_CANDIDATES)
    return f"未找到（已搜索：{searched}）"


def empty_per_source_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=["source", *ALL_METRIC_NAMES])


def empty_results_preview_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=["source", "question", "answer", *ALL_METRIC_NAMES])


def check_knowledge_base_readiness() -> Dict[str, Any]:
    markdown_dir = Path(config.MARKDOWN_DIR)
    present = {path.name for path in markdown_dir.glob("*.md")} if markdown_dir.is_dir() else set()
    found = sorted(_REQUIRED_MARKDOWN & present)
    missing = sorted(_REQUIRED_MARKDOWN - present)
    return {
        "ready": not missing,
        "found": found,
        "missing": missing,
        "total_required": len(_REQUIRED_MARKDOWN),
        "total_found": len(found),
    }


def format_kb_readiness_markdown() -> str:
    status = check_knowledge_base_readiness()
    if status["ready"]:
        docs = "、".join(f"`{name}`" for name in status["found"])
        return f"✅ **知识库就绪**（{status['total_found']}/{status['total_required']}）已包含：{docs}"
    missing = "、".join(f"`{name}`" for name in status["missing"])
    found = status["found"]
    if found:
        docs = "、".join(f"`{name}`" for name in found)
        return (
            f"⚠️ **知识库未就绪**（{status['total_found']}/{status['total_required']}）  \n"
            f"已有：{docs}  \n"
            f"缺少：{missing}  \n"
            f"请先在 **Documents** 页上传对应 PDF。"
        )
    return (
        f"⚠️ **知识库为空**  \n"
        f"缺少示例文档：{missing}  \n"
        f"请先在 **Documents** 页上传 JavaScript / Blockchain / Fortinet 示例 PDF。"
    )


def get_judge_model_name(ollama_judge_model: str = "granite4.1:3b") -> str:
    if config.LLM_PROVIDER == "deepseek":
        return config.LLM_MODEL
    return ollama_judge_model


def estimate_duration_minutes(sample_size: int, mode: str) -> str:
    count = sample_size if sample_size else 30
    if mode == "score_only":
        return f"约 {max(3, count * 2)}–{max(5, count * 4)} 分钟（每题需调用 Judge 模型 5 次）"
    if mode == "query_only":
        return f"约 {count}–{count * 2} 分钟"
    return f"约 {count * 2}–{count * 4} 分钟"


def trim_contexts_for_scoring(
    contexts: Any,
    max_items: int = 3,
    max_chars: int = 800,
) -> List[str]:
    if not isinstance(contexts, list):
        return []
    trimmed: List[str] = []
    for ctx in contexts[:max_items]:
        text = str(ctx)
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        trimmed.append(text)
    return trimmed


def trim_text_for_scoring(text: Any, max_chars: int = 600) -> str:
    value = str(text).strip()
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "…"


SCORE_METRIC_TIMEOUT_SEC = 180
JUDGE_MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", "4096"))


def _ensure_vertexai_guard() -> None:
    """RAGAS 0.4.x may import Vertex AI at startup; provide a stub when absent."""
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
    except ModuleNotFoundError:
        vertexai_module = types.ModuleType("langchain_community.chat_models.vertexai")

        class ChatVertexAI:
            pass

        vertexai_module.ChatVertexAI = ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = vertexai_module


_ensure_vertexai_guard()

from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerAccuracy,
    ContextPrecision,
    ContextRecall,
    ContextRelevance,
    ResponseGroundedness,
)


@dataclass
class EvaluationConfig:
    qa_dataset_path: Path = _DEFAULT_QA_PATH
    output_dir: Path = _DEFAULT_OUTPUT_DIR
    sample_size: Optional[int] = 5
    skip_query: bool = False
    query_only: bool = False
    dataset_csv: Optional[Path] = None
    required_markdown: frozenset[str] = _REQUIRED_MARKDOWN
    reference_context_window: int = 1400
    ollama_judge_model: str = "granite4.1:3b"
    disable_langfuse: bool = True


@dataclass
class EvaluationResult:
    ragas_df: pd.DataFrame
    scores_df: pd.DataFrame
    combined_df: pd.DataFrame
    metric_means: Dict[str, float] = field(default_factory=dict)
    per_source_means: Optional[pd.DataFrame] = None
    dataset_path: Optional[Path] = None
    results_path: Optional[Path] = None
    radar_path: Optional[Path] = None


@dataclass
class EvaluationProgress:
    phase: str
    current: int
    total: int
    message: str


@dataclass
class MetricScoreProgress:
    question_index: int
    question_total: int
    question: str
    metric_name: str
    score: float


@dataclass
class CustomMetricProgress:
    question_index: int
    question_total: int
    question: str
    hit_rate: float
    mrr: float


def format_metric_score_value(score: float) -> str:
    if score == score:
        return f"{score:.3f}"
    return "failed"


def append_metric_to_live_log(lines: List[str], event: MetricScoreProgress) -> None:
    if event.metric_name == RAGAS_METRIC_NAMES[0]:
        lines.append(f"\n### Q{event.question_index}/{event.question_total}")
        lines.append(f"> {event.question}\n")
    lines.append(f"- **{event.metric_name}**: `{format_metric_score_value(event.score)}`")


def append_custom_metrics_to_live_log(lines: List[str], event: CustomMetricProgress) -> None:
    lines.append(f"\n### Q{event.question_index}/{event.question_total} — Custom")
    lines.append(f"> {event.question}\n")
    lines.append(f"- **hit_rate**: `{format_metric_score_value(event.hit_rate)}`")
    lines.append(f"- **mrr**: `{format_metric_score_value(event.mrr)}`")


def format_metric_means(means: Dict[str, float], *, chinese: bool = False) -> str:
    if not means:
        return "_No scores yet._" if not chinese else "_暂无评分结果。_"
    sections: List[str] = []
    for section_title, metric_names in (
        ("Custom (retrieval)", CUSTOM_METRIC_NAMES),
        ("RAGAS", RAGAS_METRIC_NAMES),
    ):
        present = [name for name in metric_names if name in means]
        if not present:
            continue
        header = "| Metric | Score |" if not chinese else "| 指标 | 分数 |"
        lines = [f"**{section_title}**", header, "| --- | ---: |"]
        for name in present:
            if chinese:
                label = METRIC_LABELS_ZH.get(name, name)
            else:
                label = f"`{name}`"
            display = format_metric_score_value(means[name])
            lines.append(f"| {label} | {display} |")
        sections.append("\n".join(lines))
    return "\n\n".join(sections) if sections else ("_No scores yet._" if not chinese else "_暂无评分结果。_")


def build_radar_chart(
    metric_means: Dict[str, float],
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    present = [
        name
        for name in RAGAS_METRIC_NAMES
        if name in metric_means and metric_means[name] == metric_means[name]
    ]
    if not present:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [METRIC_LABELS_EN.get(name, name.replace("_", "\n")) for name in present]
    values = [float(metric_means[name]) for name in present]
    values_closed = values + values[:1]
    angles = np.linspace(0, 2 * np.pi, len(present), endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.plot(angles_closed, values_closed, color="steelblue", linewidth=2)
    ax.fill(angles_closed, values_closed, color="steelblue", alpha=0.25)
    ax.set_thetagrids(np.degrees(angles), labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.set_title("Overall RAGAS Profile", fontsize=13, fontweight="bold", pad=20)
    ax.grid(color="grey", linestyle="--", alpha=0.5)

    save_path = output_path or (ensure_evaluation_output_dir() / "rag_evaluation_radar.png")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def results_preview_dataframe(combined_df: pd.DataFrame) -> pd.DataFrame:
    display_cols = ["source", "question", "answer", *ALL_METRIC_NAMES]
    present = [col for col in display_cols if col in combined_df.columns]
    preview = combined_df[present].copy()
    for col in ("question", "answer"):
        if col in preview.columns:
            preview[col] = preview[col].astype(str).str.slice(0, 120)
    return preview


def parse_context_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else [value]
        except (SyntaxError, ValueError):
            return [value] if value.strip() else []
    return []


def parse_id_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except (SyntaxError, ValueError):
            pass
        return [text]
    return [str(value)]


_child_chunks_cache: Dict[str, List[Any]] = {}


def _get_child_chunks_for_markdown(markdown_file: str) -> List[Any]:
    """Return child chunks for a markdown file, reusing the ingest-time chunker output."""
    if markdown_file in _child_chunks_cache:
        return _child_chunks_cache[markdown_file]

    from document_chunker import DocumentChuncker

    md_path = Path(config.MARKDOWN_DIR) / markdown_file
    if not md_path.is_file():
        _child_chunks_cache[markdown_file] = []
        return []

    _, child_chunks = DocumentChuncker().create_chunks_single(md_path)
    _child_chunks_cache[markdown_file] = child_chunks
    return child_chunks


def clear_child_chunks_cache() -> None:
    """Drop cached chunker output (e.g. after re-ingesting documents)."""
    _child_chunks_cache.clear()


def resolve_expected_parent_ids(markdown_file: str, context_phrases: List[str]) -> List[str]:
    """Map QA context phrases to parent chunk IDs via the same chunker used at ingest."""
    child_chunks = _get_child_chunks_for_markdown(markdown_file)
    phrases = [phrase.lower() for phrase in context_phrases if phrase]
    if not phrases:
        return []

    parent_ids: set[str] = set()
    for child in child_chunks:
        content_lower = child.page_content.lower()
        if any(phrase in content_lower for phrase in phrases):
            parent_id = child.metadata.get("parent_id")
            if parent_id:
                parent_ids.add(str(parent_id))
    return sorted(parent_ids)


def extract_reference_context(
    doc_texts: Dict[str, str],
    markdown_file: str,
    phrases: List[str],
    window: int = 1400,
) -> str:
    text = doc_texts[markdown_file]
    lower_text = text.lower()
    match_positions = [
        lower_text.find(phrase.lower())
        for phrase in phrases
        if lower_text.find(phrase.lower()) >= 0
    ]
    center = min(match_positions) if match_positions else 0
    start = max(0, center - window // 3)
    end = min(len(text), center + window)
    return text[start:end].strip()


def load_markdown_corpus(required_docs: Optional[frozenset[str]] = None) -> Dict[str, str]:
    markdown_dir = Path(config.MARKDOWN_DIR)
    if not markdown_dir.is_dir():
        raise FileNotFoundError(f"Markdown directory not found: {markdown_dir}")

    md_paths = sorted(markdown_dir.glob("*.md"))
    if not md_paths:
        raise FileNotFoundError(f"No markdown files found in {markdown_dir}")

    doc_texts = {path.name: path.read_text(encoding="utf-8") for path in md_paths}
    if required_docs:
        missing = required_docs - set(doc_texts)
        if missing:
            raise FileNotFoundError(f"Missing expected markdown files: {sorted(missing)}")
    return doc_texts


DatasetLoadProgressCallback = Callable[[int, int, str], None]


def load_curated_dataset(
    qa_path: Path,
    doc_texts: Dict[str, str],
    reference_context_window: int = 1400,
    progress_callback: Optional[DatasetLoadProgressCallback] = None,
) -> List[Dict[str, Any]]:
    if not qa_path.is_file():
        raise FileNotFoundError(f"QA dataset not found: {qa_path}")

    records = json.loads(qa_path.read_text(encoding="utf-8"))
    total = len(records)

    unique_files = sorted({record["markdown_file"] for record in records})
    for index, markdown_file in enumerate(unique_files, start=1):
        if progress_callback:
            progress_callback(
                index,
                len(unique_files),
                f"[chunk] {markdown_file}",
            )
        _get_child_chunks_for_markdown(markdown_file)

    for index, record in enumerate(records, start=1):
        if progress_callback:
            progress_callback(index, total, str(record.get("question", ""))[:80])
        record["reference_contexts"] = [
            extract_reference_context(
                doc_texts,
                record["markdown_file"],
                record["context_phrases"],
                window=reference_context_window,
            )
        ]
        record["expected_parent_ids"] = resolve_expected_parent_ids(
            record["markdown_file"],
            record.get("context_phrases", []),
        )
    return records


def create_ragas_judge_llm(ollama_judge_model: str = "granite4.1:3b"):
    if config.LLM_PROVIDER == "deepseek":
        judge_model = config.LLM_MODEL
        client = AsyncOpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.DEEPSEEK_API_KEY,
        )
    else:
        judge_model = ollama_judge_model
        client = AsyncOpenAI(
            base_url="http://localhost:11434/v1/",
            api_key="ollama",
        )

    return llm_factory(
        judge_model,
        client=client,
        temperature=0,
        max_tokens=JUDGE_MAX_TOKENS,
    ), judge_model


def build_metric_scorers(ragas_llm):
    return {
        "answer_accuracy": AnswerAccuracy(llm=ragas_llm),
        "context_relevance": ContextRelevance(llm=ragas_llm),
        "response_groundedness": ResponseGroundedness(llm=ragas_llm),
        "context_precision": ContextPrecision(llm=ragas_llm),
        "context_recall": ContextRecall(llm=ragas_llm),
    }


class RAGEvaluator:
    """Run agentic RAG queries and score them with RAGAS."""

    def __init__(
        self,
        eval_config: Optional[EvaluationConfig] = None,
        rag_system: Optional[RAGSystem] = None,
    ):
        load_dotenv(_PROJECT_DIR / ".env")
        if eval_config and eval_config.disable_langfuse:
            import os

            os.environ["LANGFUSE_ENABLED"] = "false"
            config.LANGFUSE_ENABLED = False

        self.eval_config = eval_config or EvaluationConfig()
        self.eval_config.output_dir.mkdir(parents=True, exist_ok=True)

        self._rag: Optional[RAGSystem] = rag_system
        self._child_vector_store = None
        self._ragas_llm = None
        self._judge_model: Optional[str] = None

        if self._rag is not None and self._rag.agent_graph is not None:
            self._child_vector_store = self._rag.vector_db.get_collection(self._rag.collection_name)

    @property
    def dataset_csv_path(self) -> Path:
        if self.eval_config.dataset_csv:
            return self.eval_config.dataset_csv
        return self.eval_config.output_dir / "ragas_evaluation_dataset.csv"

    @property
    def results_csv_path(self) -> Path:
        return self.eval_config.output_dir / "rag_evaluation_results.csv"

    def _ensure_rag(self) -> None:
        if self._rag is not None:
            return
        self._rag = RAGSystem()
        self._rag.initialize()
        self._child_vector_store = self._rag.vector_db.get_collection(self._rag.collection_name)

    def _ensure_judge(self) -> None:
        if self._ragas_llm is None:
            self._ragas_llm, self._judge_model = create_ragas_judge_llm(
                self.eval_config.ollama_judge_model
            )

    def query_rag(self, question: str) -> Dict[str, Any]:
        self._ensure_rag()
        self._rag.reset_thread()
        graph_config = self._rag.get_config()
        result = self._rag.agent_graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            graph_config,
        )
        answer = result["messages"][-1].content
        retrieved_docs = retrieve_child_documents(
            self._child_vector_store,
            question,
            config.FUSION_TOP_K,
            enable_confidence=False,
        )
        contexts = [doc.page_content for doc in retrieved_docs]
        retrieved_parent_ids = extract_ids_from_documents(retrieved_docs)
        return {
            "answer": answer,
            "contexts": contexts,
            "retrieved_parent_ids": retrieved_parent_ids,
        }

    def _retrieve_parent_ids(self, question: str) -> List[str]:
        self._ensure_rag()
        retrieved_docs = retrieve_child_documents(
            self._child_vector_store,
            question,
            config.FUSION_TOP_K,
            enable_confidence=False,
        )
        return extract_ids_from_documents(retrieved_docs)

    def apply_custom_metrics(
        self,
        df: pd.DataFrame,
        expected_by_question: Optional[Dict[str, List[str]]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> pd.DataFrame:
        """Compute hit_rate and mrr for each row; fill missing parent-id columns if needed."""
        enriched = df.copy()
        if "expected_parent_ids" not in enriched.columns:
            enriched["expected_parent_ids"] = [[] for _ in range(len(enriched))]
        if "retrieved_parent_ids" not in enriched.columns:
            enriched["retrieved_parent_ids"] = [[] for _ in range(len(enriched))]

        enriched["expected_parent_ids"] = enriched["expected_parent_ids"].apply(parse_id_list)
        enriched["retrieved_parent_ids"] = enriched["retrieved_parent_ids"].apply(parse_id_list)

        hit_rates: List[float] = []
        mrr_values: List[float] = []
        total = len(enriched)

        for step, (row_index, row) in enumerate(enriched.iterrows(), start=1):
            question = str(row["user_input"])
            expected_ids = list(row["expected_parent_ids"])
            if not expected_ids and expected_by_question:
                expected_ids = expected_by_question.get(question, [])

            retrieved_ids = list(row["retrieved_parent_ids"])
            if not retrieved_ids:
                try:
                    retrieved_ids = self._retrieve_parent_ids(question)
                    enriched.at[row_index, "retrieved_parent_ids"] = retrieved_ids
                except Exception as exc:
                    logger.warning("Custom metric retrieval failed for '%s': %s", question[:60], exc)
                    retrieved_ids = []

            custom_scores = evaluate_custom_metrics(retrieved_ids, expected_ids)
            hit_rates.append(custom_scores["hit_rate"])
            mrr_values.append(custom_scores["mrr"])

            if progress_callback:
                progress_callback(
                    CustomMetricProgress(
                        question_index=step,
                        question_total=total,
                        question=question[:120],
                        hit_rate=custom_scores["hit_rate"],
                        mrr=custom_scores["mrr"],
                    )
                )

        enriched["hit_rate"] = hit_rates
        enriched["mrr"] = mrr_values
        return enriched

    def run_queries(
        self,
        records: List[Dict[str, Any]],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[Dict[str, Any]]:
        sample_size = self.eval_config.sample_size
        eval_records = records[:sample_size] if sample_size else records
        total = len(eval_records)
        logger.info("Querying RAG for %s / %s questions", total, len(records))

        results: List[Dict[str, Any]] = []
        use_tqdm = progress_callback is None and total > 1
        iterator = tqdm(eval_records, total=total, desc="Querying RAG", disable=not use_tqdm)
        for index, record in enumerate(iterator, start=1):
            question = record["question"]
            expected_parent_ids = record.get("expected_parent_ids", [])
            if progress_callback:
                progress_callback(
                    EvaluationProgress(
                        phase="query",
                        current=index,
                        total=total,
                        message=f"Querying RAG ({index}/{total}): {question[:80]}",
                    )
                )

            try:
                rag_output = self.query_rag(question)
                custom_scores = evaluate_custom_metrics(
                    rag_output.get("retrieved_parent_ids", []),
                    expected_parent_ids,
                )
                if progress_callback:
                    progress_callback(
                        CustomMetricProgress(
                            question_index=index,
                            question_total=total,
                            question=question[:120],
                            hit_rate=custom_scores["hit_rate"],
                            mrr=custom_scores["mrr"],
                        )
                    )
                results.append(
                    {
                        "source": record["source"],
                        "question": question,
                        "answer": rag_output["answer"],
                        "contexts": rag_output["contexts"],
                        "ground_truth": record["reference"],
                        "reference_contexts": record["reference_contexts"],
                        "expected_parent_ids": expected_parent_ids,
                        "retrieved_parent_ids": rag_output.get("retrieved_parent_ids", []),
                        "hit_rate": custom_scores["hit_rate"],
                        "mrr": custom_scores["mrr"],
                    }
                )
            except Exception as exc:
                logger.exception("RAG query failed for: %s", question[:80])
                results.append(
                    {
                        "source": record["source"],
                        "question": question,
                        "answer": "",
                        "contexts": [],
                        "ground_truth": record["reference"],
                        "reference_contexts": record["reference_contexts"],
                        "expected_parent_ids": expected_parent_ids,
                        "retrieved_parent_ids": [],
                        "hit_rate": 0.0,
                        "mrr": 0.0,
                        "error": str(exc),
                    }
                )
        return results

    @staticmethod
    def results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "source": [r.get("source", "unknown") for r in results],
                "user_input": [r["question"] for r in results],
                "retrieved_contexts": [r["contexts"] for r in results],
                "reference_contexts": [r.get("reference_contexts", []) for r in results],
                "response": [r["answer"] for r in results],
                "reference": [r["ground_truth"] for r in results],
                "expected_parent_ids": [r.get("expected_parent_ids", []) for r in results],
                "retrieved_parent_ids": [r.get("retrieved_parent_ids", []) for r in results],
                "hit_rate": [r.get("hit_rate", float("nan")) for r in results],
                "mrr": [r.get("mrr", float("nan")) for r in results],
            }
        )

    def save_dataset(self, ragas_df: pd.DataFrame) -> Path:
        path = self.dataset_csv_path
        ragas_df.to_csv(path, index=False)
        logger.info("Saved RAG outputs to %s", path)
        return path

    def load_dataset(self, path: Optional[Path] = None) -> pd.DataFrame:
        csv_path = path or self.dataset_csv_path
        if not csv_path.is_file():
            raise FileNotFoundError(f"Evaluation dataset CSV not found: {csv_path}")
        ragas_df = pd.read_csv(csv_path)
        ragas_df["retrieved_contexts"] = ragas_df["retrieved_contexts"].apply(parse_context_list)
        if "reference_contexts" in ragas_df.columns:
            ragas_df["reference_contexts"] = ragas_df["reference_contexts"].apply(parse_context_list)
        if "expected_parent_ids" in ragas_df.columns:
            ragas_df["expected_parent_ids"] = ragas_df["expected_parent_ids"].apply(parse_id_list)
        if "retrieved_parent_ids" in ragas_df.columns:
            ragas_df["retrieved_parent_ids"] = ragas_df["retrieved_parent_ids"].apply(parse_id_list)
        return ragas_df

    async def score_row(
        self,
        row: pd.Series,
        scorers: Dict[str, Any],
        progress_callback: Optional[ProgressCallback] = None,
        question_index: int = 1,
        question_total: int = 1,
    ) -> Dict[str, float]:
        retrieved_contexts = trim_contexts_for_scoring(row["retrieved_contexts"])
        reference = trim_text_for_scoring(row["reference"])
        response = trim_text_for_scoring(row["response"], max_chars=1200)
        question = str(row["user_input"])[:120]
        metric_inputs = {
            "answer_accuracy": {
                "user_input": row["user_input"],
                "response": response,
                "reference": reference,
            },
            "context_relevance": {
                "user_input": row["user_input"],
                "retrieved_contexts": retrieved_contexts,
            },
            "response_groundedness": {
                "response": response,
                "retrieved_contexts": retrieved_contexts,
            },
            "context_precision": {
                "user_input": row["user_input"],
                "reference": reference,
                "retrieved_contexts": retrieved_contexts,
            },
            "context_recall": {
                "user_input": row["user_input"],
                "retrieved_contexts": retrieved_contexts,
                "reference": reference,
            },
        }

        async def score_metric(metric_name: str, scorer: Any) -> Tuple[str, float]:
            try:
                result = await asyncio.wait_for(
                    scorer.ascore(**metric_inputs[metric_name]),
                    timeout=SCORE_METRIC_TIMEOUT_SEC,
                )
                return metric_name, float(result.value)
            except Exception as exc:
                logger.warning("Metric %s failed: %s", metric_name, exc)
                return metric_name, float("nan")

        scores: Dict[str, float] = {}
        for metric_name in RAGAS_METRIC_NAMES:
            if metric_name not in scorers:
                continue
            name, value = await score_metric(metric_name, scorers[metric_name])
            scores[name] = value
            if progress_callback:
                progress_callback(
                    MetricScoreProgress(
                        question_index=question_index,
                        question_total=question_total,
                        question=question,
                        metric_name=name,
                        score=value,
                    )
                )
        return scores

    async def score_dataframe(
        self,
        ragas_df: pd.DataFrame,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> pd.DataFrame:
        self._ensure_judge()
        scorers = build_metric_scorers(self._ragas_llm)
        score_rows: List[Dict[str, float]] = []
        total = len(ragas_df)
        use_tqdm = progress_callback is None and total > 1

        for step, (idx, row) in enumerate(
            tqdm(ragas_df.iterrows(), total=total, desc="Evaluating", disable=not use_tqdm),
            start=1,
        ):
            question = str(row["user_input"])[:90]
            if progress_callback:
                progress_callback(
                    EvaluationProgress(
                        phase="score",
                        current=step,
                        total=total,
                        message=f"RAGAS scoring ({step}/{total}): {question}",
                    )
                )
            else:
                logger.info("[%s/%s] %s", step, total, question)
            score_rows.append(
                await self.score_row(
                    row,
                    scorers,
                    progress_callback=progress_callback,
                    question_index=step,
                    question_total=total,
                )
            )
        return pd.DataFrame(score_rows)

    @staticmethod
    def combine_results(ragas_df: pd.DataFrame, scores_df: pd.DataFrame) -> pd.DataFrame:
        custom_cols = [col for col in (*CUSTOM_METRIC_NAMES, "expected_parent_ids", "retrieved_parent_ids") if col in ragas_df.columns]
        ragas_base = ragas_df.drop(columns=custom_cols, errors="ignore")
        df = pd.concat([ragas_base.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)
        for col in custom_cols:
            df[col] = ragas_df[col].reset_index(drop=True)
        df["question"] = df["user_input"]
        df["answer"] = df["response"]
        df["ground_truth"] = df["reference"]
        return df

    @staticmethod
    def aggregate_means(combined_df: pd.DataFrame) -> Dict[str, float]:
        present = [name for name in ALL_METRIC_NAMES if name in combined_df.columns]
        if not present:
            return {}
        means = combined_df[present].mean(numeric_only=True).round(3)
        return {key: float(value) for key, value in means.items()}

    @staticmethod
    def aggregate_by_source(combined_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if "source" not in combined_df.columns:
            return None
        present = [name for name in ALL_METRIC_NAMES if name in combined_df.columns]
        if not present:
            return None
        return combined_df.groupby("source")[present].mean(numeric_only=True).round(3)

    def run(
        self,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> EvaluationResult:
        result: Optional[EvaluationResult] = None
        for event in self.iter_run(progress_callback=progress_callback):
            if isinstance(event, EvaluationResult):
                result = event
        if result is None:
            raise RuntimeError("Evaluation finished without a result.")
        return result

    def _build_expected_parent_map(self) -> Dict[str, List[str]]:
        try:
            doc_texts = load_markdown_corpus(self.eval_config.required_markdown or None)
            records = load_curated_dataset(
                self.eval_config.qa_dataset_path,
                doc_texts,
                reference_context_window=self.eval_config.reference_context_window,
            )
            return {record["question"]: record.get("expected_parent_ids", []) for record in records}
        except Exception as exc:
            logger.warning("Could not build expected parent-id map: %s", exc)
            return {}

    def _needs_custom_metric_pass(self, df: pd.DataFrame) -> bool:
        if "hit_rate" not in df.columns or "mrr" not in df.columns:
            return True
        return bool(df[["hit_rate", "mrr"]].isna().any().any())

    def iter_run(
        self,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        def emit(event: EvaluationProgress):
            if progress_callback:
                progress_callback(event)
            return event

        yield emit(
            EvaluationProgress(phase="prepare", current=0, total=1, message="正在加载评测数据集...")
        )

        dataset_path: Optional[Path] = None
        results_path: Optional[Path] = None

        if self.eval_config.skip_query:
            ragas_df = self.load_dataset()
            if self.eval_config.sample_size:
                ragas_df = ragas_df.head(self.eval_config.sample_size).reset_index(drop=True)
            dataset_path = self.dataset_csv_path
            if self._needs_custom_metric_pass(ragas_df):
                yield emit(
                    EvaluationProgress(
                        phase="score",
                        current=0,
                        total=len(ragas_df),
                        message="计算 Custom 指标（hit_rate / mrr）…",
                    )
                )
                ragas_df = self.apply_custom_metrics(
                    ragas_df,
                    expected_by_question=self._build_expected_parent_map(),
                    progress_callback=progress_callback,
                )
        else:
            doc_texts = load_markdown_corpus(self.eval_config.required_markdown or None)

            def on_dataset_load(current: int, total: int, detail: str) -> None:
                if detail.startswith("[chunk]"):
                    markdown_file = detail.removeprefix("[chunk] ").strip()
                    message = (
                        f"正在预计算文档分块 ({current}/{total})：{markdown_file} "
                        "（首次约需 30–60 秒/篇，请稍候）"
                    )
                else:
                    message = f"正在加载评测数据集 ({current}/{total})…"
                emit(
                    EvaluationProgress(
                        phase="prepare",
                        current=current,
                        total=total,
                        message=message,
                    )
                )

            records = load_curated_dataset(
                self.eval_config.qa_dataset_path,
                doc_texts,
                reference_context_window=self.eval_config.reference_context_window,
                progress_callback=on_dataset_load,
            )
            query_results = self.run_queries(records, progress_callback=progress_callback)
            ragas_df = self.results_to_dataframe(query_results)
            dataset_path = self.save_dataset(ragas_df)

        if self.eval_config.query_only:
            metric_means = self.aggregate_means(ragas_df)
            yield EvaluationResult(
                ragas_df=ragas_df,
                scores_df=pd.DataFrame(),
                combined_df=ragas_df.copy(),
                metric_means=metric_means,
                per_source_means=self.aggregate_by_source(ragas_df),
                dataset_path=dataset_path,
            )
            return

        yield emit(
            EvaluationProgress(
                phase="score",
                current=0,
                total=len(ragas_df),
                message="开始 RAGAS 打分…（首题约需 1–3 分钟，请勿关闭页面）",
            )
        )

        scores_df = asyncio.run(
            self.score_dataframe(ragas_df, progress_callback=progress_callback)
        )
        combined_df = self.combine_results(ragas_df, scores_df)
        combined_df.to_csv(self.results_csv_path, index=False)
        results_path = self.results_csv_path
        metric_means = self.aggregate_means(combined_df)
        radar_path = build_radar_chart(
            {name: value for name, value in metric_means.items() if name in RAGAS_METRIC_NAMES},
            self.eval_config.output_dir / "rag_evaluation_radar.png",
        )

        yield emit(
            EvaluationProgress(phase="done", current=1, total=1, message="评估完成。")
        )

        yield EvaluationResult(
            ragas_df=ragas_df,
            scores_df=scores_df,
            combined_df=combined_df,
            metric_means=metric_means,
            per_source_means=self.aggregate_by_source(combined_df),
            dataset_path=dataset_path,
            results_path=results_path,
            radar_path=radar_path,
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Agentic RAG with RAGAS metrics.")
    parser.add_argument(
        "--sample",
        type=int,
        default=5,
        help="Number of QA pairs to run (default: 5). Use 0 for all 30.",
    )
    parser.add_argument(
        "--qa-dataset",
        type=Path,
        default=_DEFAULT_QA_PATH,
        help="Path to curated_ragas_qa.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory for evaluation CSV outputs",
    )
    parser.add_argument(
        "--skip-query",
        action="store_true",
        help="Skip RAG queries; score an existing ragas_evaluation_dataset.csv",
    )
    parser.add_argument(
        "--query-only",
        action="store_true",
        help="Only run RAG queries and save the dataset CSV",
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=None,
        help="Override path to ragas_evaluation_dataset.csv",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)

    eval_config = EvaluationConfig(
        qa_dataset_path=args.qa_dataset,
        output_dir=args.output_dir,
        sample_size=None if args.sample == 0 else args.sample,
        skip_query=args.skip_query,
        query_only=args.query_only,
        dataset_csv=args.dataset_csv,
    )

    evaluator = RAGEvaluator(eval_config)
    print(f"LLM provider: {config.LLM_PROVIDER}")
    print(f"Answer model: {config.LLM_MODEL}")

    result = evaluator.run()

    if result.metric_means:
        print("\nMean scores:")
        for name, value in result.metric_means.items():
            print(f"  {name}: {value:.3f}")

    if result.per_source_means is not None:
        print("\nMean scores per source:")
        print(result.per_source_means.to_string())

    print(f"\nDataset: {evaluator.dataset_csv_path}")
    if not eval_config.query_only:
        print(f"Results: {evaluator.results_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
