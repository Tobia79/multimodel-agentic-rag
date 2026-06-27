from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import gradio as gr
import pandas as pd

import config
from core.evaluation import (
    EvaluationConfig,
    EvaluationProgress,
    EvaluationResult,
    MetricScoreProgress,
    CustomMetricProgress,
    RAGEvaluator,
    append_custom_metrics_to_live_log,
    append_metric_to_live_log,
    display_repo_path,
    empty_per_source_dataframe,
    empty_results_preview_dataframe,
    ensure_evaluation_output_dir,
    estimate_duration_minutes,
    find_dataset_csv,
    format_dataset_status,
    format_kb_readiness_markdown,
    format_metric_means,
    format_output_dir_line,
    format_confidence_config_line,
    format_retrieval_config_line,
    format_rerank_config_line,
    get_judge_model_name,
    results_preview_dataframe,
)

EVAL_MODES = {
    "full": "完整评估（查询 + RAGAS 打分）",
    "query_only": "仅查询（保存 dataset CSV）",
    "score_only": "仅打分（读取已有 dataset CSV）",
}

GradioEvalUpdate = Tuple[
    str,
    str,
    str,
    pd.DataFrame,
    pd.DataFrame,
    gr.update,
    Optional[str],
    Optional[str],
    gr.update,
    gr.update,
    gr.update,
]

_EMPTY_LIVE_LOG = "## Live Scores\n\n_Waiting to start…_"
_HIDDEN_RADAR = gr.update(visible=False, value=None)


def _format_progress_status(event: EvaluationProgress) -> str:
    phase_label = {
        "prepare": "准备",
        "query": "RAG 查询",
        "score": "RAGAS / Custom 打分",
        "done": "完成",
    }.get(event.phase, event.phase)

    if event.phase == "done":
        return f"✅ **{phase_label}**"
    if event.total > 0 and event.phase in {"prepare", "query", "score"}:
        return f"⏳ **{phase_label}** ({event.current}/{event.total})  \n{event.message}"
    return f"⏳ **{phase_label}**  \n{event.message}"


def _format_live_log(lines: List[str]) -> str:
    if not lines:
        return _EMPTY_LIVE_LOG
    return "## Live Scores\n" + "\n".join(lines)


def _empty_yield(
    status: str,
    live_log: str,
    metric_summary: str,
    button: gr.update,
    *,
    show_results: gr.update,
    show_downloads: gr.update,
    radar_update: gr.update,
) -> GradioEvalUpdate:
    empty_per_source = empty_per_source_dataframe()
    empty_results = empty_results_preview_dataframe()
    return (
        status,
        live_log,
        metric_summary,
        empty_per_source,
        empty_results,
        radar_update,
        None,
        None,
        show_results,
        show_downloads,
        button,
    )


_THREAD_SENTINEL = object()


def _run_evaluator_worker(
    evaluator: RAGEvaluator,
    on_progress,
    event_queue: queue.Queue,
) -> None:
    try:
        for event in evaluator.iter_run(progress_callback=on_progress):
            if isinstance(event, EvaluationResult):
                event_queue.put(event)
    except Exception as exc:
        event_queue.put(exc)
    finally:
        event_queue.put(_THREAD_SENTINEL)


class EvaluationInterface:
    """Gradio-facing wrapper around :class:`RAGEvaluator`."""

    def __init__(self, rag_system):
        self.rag_system = rag_system
        ensure_evaluation_output_dir()

    @staticmethod
    def kb_status() -> str:
        return format_kb_readiness_markdown()

    @staticmethod
    def can_start(mode: str, dataset_csv: Optional[str] = None) -> Tuple[bool, str]:
        if mode == "score_only":
            custom = Path(dataset_csv) if dataset_csv else None
            if find_dataset_csv(custom):
                return True, ""
            return (
                False,
                f"未找到 dataset：{format_dataset_status(custom)}。"
                "可上传 CSV，或先运行「完整评估」/「仅查询」。",
            )
        if check_knowledge_base_ready():
            return True, ""
        return False, "知识库未包含全部示例文档，请先在 Documents 页上传对应 PDF。"

    @staticmethod
    def config_summary(
        sample_size: int,
        mode: str,
        dataset_csv: Optional[str] = None,
    ) -> str:
        ensure_evaluation_output_dir()
        judge = get_judge_model_name()
        duration = estimate_duration_minutes(sample_size, mode)
        custom = Path(dataset_csv) if dataset_csv else None
        lines = [
            f"- **回答模型**：`{config.LLM_PROVIDER}` / `{config.LLM_MODEL}`",
            f"- **RAGAS Judge**：`{judge}`",
            format_retrieval_config_line(),
            format_confidence_config_line(),
            format_rerank_config_line(),
            "- **Custom 指标**：`Hit_rate`、`MRR`",
            f"- **题量**：{sample_size if sample_size else 30} 题",
            f"- **模式**：{EVAL_MODES.get(mode, mode)}",
            f"- **预计耗时**：{duration}",
            format_output_dir_line(),
        ]
        if mode == "score_only":
            resolved = find_dataset_csv(custom)
            if resolved:
                lines.append(f"- **Dataset**：`{display_repo_path(resolved)}`")
            else:
                lines.append(f"- **Dataset**：{format_dataset_status(custom)}")
        return "  \n".join(lines)

    def run(
        self,
        sample_size: int,
        mode: str,
        progress=None,
        dataset_csv: Optional[str] = None,
    ) -> Generator[GradioEvalUpdate, None, None]:
        skip_query = mode == "score_only"
        query_only = mode == "query_only"
        hidden_results = gr.update(visible=False)
        hidden_downloads = gr.update(visible=False)
        disabled_btn = gr.update(interactive=False, value="评估中…")
        enabled_btn = gr.update(interactive=True, value="开始评估")

        can_run, block_reason = self.can_start(mode, dataset_csv=dataset_csv)

        if not can_run:
            yield _empty_yield(
                f"❌ **无法开始**：{block_reason}",
                _EMPTY_LIVE_LOG,
                "_No scores yet._",
                enabled_btn,
                show_results=hidden_results,
                show_downloads=hidden_downloads,
                radar_update=_HIDDEN_RADAR,
            )
            return

        resolved_dataset = find_dataset_csv(Path(dataset_csv) if dataset_csv else None)
        eval_config = EvaluationConfig(
            sample_size=None if sample_size == 0 else sample_size,
            skip_query=skip_query,
            query_only=query_only,
            dataset_csv=resolved_dataset,
            disable_langfuse=True,
        )
        evaluator = RAGEvaluator(eval_config, rag_system=self.rag_system)
        event_queue: queue.Queue = queue.Queue()
        live_log_lines: List[str] = []

        def on_progress(event) -> None:
            event_queue.put(event)
            if progress is None or not isinstance(event, EvaluationProgress):
                return
            if event.phase == "prepare":
                ratio = event.current / event.total if event.total > 0 else 0.0
                progress(min(ratio * 0.1, 0.09), desc=event.message)
                return
            if event.phase == "done":
                progress(1.0, desc=event.message)
                return
            if event.total > 0:
                ratio = event.current / event.total
                if event.phase == "score" and mode == "full":
                    ratio = 0.5 + (ratio * 0.5)
                elif event.phase == "query":
                    ratio = ratio * (0.5 if mode == "full" else 1.0)
                progress(min(ratio, 0.99), desc=event.message)

        yield _empty_yield(
            "⏳ **准备中…**",
            "## Live Scores\n\n_Running…_",
            "_Scoring in progress…_",
            disabled_btn,
            show_results=hidden_results,
            show_downloads=hidden_downloads,
            radar_update=_HIDDEN_RADAR,
        )

        worker = threading.Thread(
            target=_run_evaluator_worker,
            args=(evaluator, on_progress, event_queue),
            daemon=True,
        )
        worker.start()

        result: Optional[EvaluationResult] = None
        try:
            while True:
                try:
                    item = event_queue.get(timeout=1.0)
                except queue.Empty:
                    if not worker.is_alive():
                        break
                    continue

                if item is _THREAD_SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item
                if isinstance(item, MetricScoreProgress):
                    append_metric_to_live_log(live_log_lines, item)
                    yield _empty_yield(
                        _format_progress_status(
                            EvaluationProgress(
                                phase="score",
                                current=item.question_index,
                                total=item.question_total,
                                message=f"RAGAS scoring ({item.question_index}/{item.question_total})",
                            )
                        ),
                        _format_live_log(live_log_lines),
                        "_Scoring in progress…_",
                        disabled_btn,
                        show_results=gr.update(visible=True, open=True),
                        show_downloads=hidden_downloads,
                        radar_update=_HIDDEN_RADAR,
                    )
                elif isinstance(item, CustomMetricProgress):
                    append_custom_metrics_to_live_log(live_log_lines, item)
                    phase = "query" if mode == "full" else "score"
                    yield _empty_yield(
                        _format_progress_status(
                            EvaluationProgress(
                                phase=phase,
                                current=item.question_index,
                                total=item.question_total,
                                message=(
                                    f"Custom metrics ({item.question_index}/{item.question_total})"
                                ),
                            )
                        ),
                        _format_live_log(live_log_lines),
                        "_Computing custom metrics…_" if phase == "score" else "_Waiting…_",
                        disabled_btn,
                        show_results=gr.update(visible=True, open=True),
                        show_downloads=hidden_downloads,
                        radar_update=_HIDDEN_RADAR,
                    )
                elif isinstance(item, EvaluationProgress):
                    if item.phase == "score" and not query_only:
                        if not live_log_lines:
                            live_log_lines.append("\n_Scoring started…_\n")
                    yield _empty_yield(
                        _format_progress_status(item),
                        _format_live_log(live_log_lines) if live_log_lines else "## Live Scores\n\n_Running…_",
                        "_Scoring in progress…_" if item.phase == "score" else "_Waiting…_",
                        disabled_btn,
                        show_results=gr.update(visible=True, open=True) if item.phase == "score" else hidden_results,
                        show_downloads=hidden_downloads,
                        radar_update=_HIDDEN_RADAR,
                    )
                elif isinstance(item, EvaluationResult):
                    result = item
        except FileNotFoundError as exc:
            yield _empty_yield(
                f"❌ **错误**：{exc}",
                _format_live_log(live_log_lines),
                "_Scoring failed._",
                enabled_btn,
                show_results=hidden_results,
                show_downloads=hidden_downloads,
                radar_update=_HIDDEN_RADAR,
            )
            return
        except Exception as exc:
            yield _empty_yield(
                f"❌ **错误**：{exc}",
                _format_live_log(live_log_lines),
                "_Scoring failed._",
                enabled_btn,
                show_results=hidden_results,
                show_downloads=hidden_downloads,
                radar_update=_HIDDEN_RADAR,
            )
            return

        worker.join(timeout=5)

        if result is None:
            yield _empty_yield(
                "❌ **错误**：评估未返回结果。",
                _format_live_log(live_log_lines),
                "_Scoring failed._",
                enabled_btn,
                show_results=hidden_results,
                show_downloads=hidden_downloads,
                radar_update=_HIDDEN_RADAR,
            )
            return

        empty_per_source = empty_per_source_dataframe()
        if query_only:
            means_md = format_metric_means(result.metric_means, chinese=False)
            per_source = result.per_source_means if result.per_source_means is not None else empty_per_source
            preview = results_preview_dataframe(result.combined_df)
            final_status = f"✅ **完成**：已保存 {len(result.ragas_df)} 条查询结果（含 Custom 指标）。"
            radar_path = None
        else:
            means_md = format_metric_means(result.metric_means, chinese=False)
            per_source = result.per_source_means if result.per_source_means is not None else empty_per_source
            preview = results_preview_dataframe(result.combined_df)
            if skip_query:
                final_status = f"✅ **完成**：已对 {len(result.combined_df)} 条记录打分。"
            else:
                final_status = f"✅ **完成**：共评估 {len(result.combined_df)} 题。"
            radar_path = (
                str(result.radar_path)
                if result.radar_path and Path(result.radar_path).is_file()
                else None
            )

        dataset_file = (
            str(result.dataset_path)
            if result.dataset_path and Path(result.dataset_path).is_file()
            else None
        )
        results_file = (
            str(result.results_path)
            if result.results_path and Path(result.results_path).is_file()
            else None
        )

        yield (
            final_status,
            _format_live_log(live_log_lines) if live_log_lines else "## Live Scores\n\n_Completed._",
            means_md,
            per_source,
            preview,
            gr.update(visible=bool(radar_path), value=radar_path),
            dataset_file,
            results_file,
            gr.update(visible=True, open=True),
            gr.update(visible=bool(dataset_file or results_file)),
            enabled_btn,
        )


def check_knowledge_base_ready() -> bool:
    from core.evaluation import check_knowledge_base_readiness

    return check_knowledge_base_readiness()["ready"]
