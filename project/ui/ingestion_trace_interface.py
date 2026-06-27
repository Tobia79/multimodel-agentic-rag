"""Gradio UI helpers for browsing ingestion trace history."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config
import pandas as pd


MAIN_STAGES = ("load", "split", "transform", "embed", "upsert")
STAGE_LABELS = {
    "load": "📄 Load",
    "split": "✂️ Split",
    "transform": "🔄 Transform",
    "embed": "🔢 Embed",
    "upsert": "💾 Upsert",
    "chunk_refiner": "🧹 Chunk Refiner",
    "metadata_enricher": "🏷️ Metadata Enricher",
    "image_captioner": "🖼️ Image Captioner",
}

IMAGE_REF_PATTERN = re.compile(r"\[IMAGE:\s*([^\]]+)\]")


def _basename(path: str) -> str:
    if not path:
        return "—"
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def _empty_timings_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["Stage", "Elapsed (ms)"])


def _json_pretty(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _split_body_and_image_lines(text: str) -> Tuple[str, str]:
    if not text:
        return "", ""
    body_lines: List[str] = []
    image_lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if IMAGE_REF_PATTERN.search(line) or stripped.startswith("(Description:"):
            image_lines.append(line)
        else:
            body_lines.append(line)
    return "\n".join(body_lines).strip(), "\n".join(image_lines).strip()


def _metadata_without_images(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not metadata:
        return {}
    return {key: value for key, value in metadata.items() if key != "images"}


class IngestionTraceInterface:
    def __init__(self, doc_manager):
        self.doc_manager = doc_manager
        self._traces: List[Dict[str, Any]] = []
        self._split_parents: List[Dict[str, Any]] = []
        self._split_children: List[Dict[str, Any]] = []
        self._transform_chunks: List[Dict[str, Any]] = []
        self._embed_chunks: List[Dict[str, Any]] = []

    def refresh_traces(self, limit: int = 50) -> None:
        self._traces = self.doc_manager.list_ingestion_traces(limit=limit)

    def trace_choices(self) -> List[str]:
        choices = []
        for trace in self._traces:
            meta = trace.get("metadata", {})
            source = _basename(meta.get("source_path", "unknown"))
            started = (trace.get("started_at") or "—")[:19]
            elapsed = trace.get("elapsed_ms")
            elapsed_label = f"{elapsed:.0f} ms" if isinstance(elapsed, (int, float)) else "—"
            status = "✅" if trace.get("success") else "❌"
            choices.append(f"{status} {source} · {elapsed_label} · {started}")
        return choices

    def _get_trace(self, choice: Optional[str]) -> Optional[Dict[str, Any]]:
        if not choice or not self._traces:
            return None
        try:
            index = self.trace_choices().index(choice)
        except ValueError:
            return None
        return self._traces[index]

    @staticmethod
    def _stages_by_name(trace: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        mapping: Dict[str, Dict[str, Any]] = {}
        for stage in trace.get("stages", []):
            name = stage.get("stage")
            if name:
                mapping[name] = stage
        return mapping

    def _cache_trace_chunks(self, trace: Dict[str, Any]) -> None:
        stages = self._stages_by_name(trace)
        split_d = stages.get("split", {}).get("data", {})
        self._split_parents = split_d.get("parents", [])
        self._split_children = split_d.get("chunks", [])
        self._transform_chunks = stages.get("transform", {}).get("data", {}).get("chunks", [])
        self._embed_chunks = stages.get("embed", {}).get("data", {}).get("chunks", [])

    @staticmethod
    def _choice_options(items: List[Dict[str, Any]], label_fn) -> List[Tuple[str, str]]:
        return [(label_fn(index, item), str(index)) for index, item in enumerate(items)]

    @staticmethod
    def _resolve_index(choice_value: Optional[str], size: int) -> int:
        if not choice_value or size == 0:
            return 0
        try:
            index = int(choice_value)
            return max(0, min(index, size - 1))
        except ValueError:
            return 0

    def _split_parent_index_df(self) -> pd.DataFrame:
        rows = []
        for parent in self._split_parents:
            rows.append(
                {
                    "#": parent.get("parent_index", 0) + 1,
                    "Parent ID": parent.get("parent_id", ""),
                    "Chars": parent.get("char_len", 0),
                }
            )
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["#", "Parent ID", "Chars"])

    def _split_child_index_df(self) -> pd.DataFrame:
        rows = []
        for chunk in self._split_children:
            refs = chunk.get("image_refs") or []
            rows.append(
                {
                    "#": chunk.get("chunk_index", 0) + 1,
                    "Parent ID": chunk.get("parent_id", ""),
                    "Chars": chunk.get("char_len", 0),
                    "Image Refs": len(refs),
                }
            )
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["#", "Parent ID", "Chars", "Image Refs"]
        )

    def render_split_parent(self, choice_value: Optional[str] = None) -> Tuple[str, str, str, str]:
        if not self._split_parents:
            return "_无父块记录_", "", "", ""
        index = self._resolve_index(choice_value, len(self._split_parents))
        parent = self._split_parents[index]
        info = (
            f"**Parent #{parent.get('parent_index', index) + 1}** · "
            f"`{parent.get('parent_id', '')}` · {parent.get('char_len', 0):,} chars"
        )
        metadata = _json_pretty(_metadata_without_images(parent.get("metadata")))
        text = parent.get("text", "")
        body, image_lines = _split_body_and_image_lines(text)
        images = image_lines or "_该父块正文中无图片占位符_"
        return info, metadata, body or text, images

    def render_split_child(self, choice_value: Optional[str] = None) -> Tuple[str, str, str, str, str]:
        if not self._split_children:
            return "_无子块记录_", "", "", "", ""
        index = self._resolve_index(choice_value, len(self._split_children))
        chunk = self._split_children[index]
        refs = chunk.get("image_refs") or []
        info = (
            f"**Child #{chunk.get('chunk_index', index) + 1}** · "
            f"`{chunk.get('parent_id', '')}` · {chunk.get('char_len', 0):,} chars · "
            f"图片引用 {len(refs)} 个"
        )
        metadata = _json_pretty(_metadata_without_images(chunk.get("metadata")))
        text_body = chunk.get("text_body")
        if text_body is None:
            text_body, _ = _split_body_and_image_lines(chunk.get("text", ""))
        image_lines = chunk.get("image_lines")
        if image_lines is None:
            _, image_lines = _split_body_and_image_lines(chunk.get("text", ""))
        images_in_chunk = chunk.get("images_in_chunk") or []
        images_meta = _json_pretty(images_in_chunk) if images_in_chunk else "_该子块未引用图片_"
        return info, metadata, text_body, image_lines or "_无图片占位符行_", images_meta

    def _transform_index_df(self) -> pd.DataFrame:
        rows = []
        for chunk in self._transform_chunks:
            rows.append(
                {
                    "#": chunk.get("chunk_index", 0) + 1,
                    "Parent ID": chunk.get("parent_id", ""),
                    "Chars": chunk.get("char_len", 0),
                    "Refined": chunk.get("refined_by", ""),
                    "Enriched": chunk.get("enriched_by", ""),
                    "Title": chunk.get("title", ""),
                }
            )
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["#", "Parent ID", "Chars", "Refined", "Enriched", "Title"]
        )

    def render_transform_chunk(self, choice_value: Optional[str] = None) -> Tuple[str, str, str, str, str, str, str]:
        if not self._transform_chunks:
            return "_无 Transform 记录_", "", "", "", "", "", ""
        index = self._resolve_index(choice_value, len(self._transform_chunks))
        chunk = self._transform_chunks[index]
        tags = ", ".join(str(tag) for tag in chunk.get("tags", [])) or "—"
        info = (
            f"**Chunk #{chunk.get('chunk_index', index) + 1}** · "
            f"`{chunk.get('parent_id', '')}` · {chunk.get('char_len', 0):,} chars\n\n"
            f"- **Refined By**: `{chunk.get('refined_by', '—')}`\n"
            f"- **Enriched By**: `{chunk.get('enriched_by', '—')}`\n"
            f"- **Title**: {chunk.get('title', '—')}\n"
            f"- **Summary**: {chunk.get('summary', '—')}\n"
            f"- **Tags**: {tags}"
        )
        text_before = chunk.get("text_before", "")
        text_after_refine = chunk.get("text_after_refine", text_before)
        text_after_enrich = chunk.get("text_after_enrich", text_after_refine)
        text_after = chunk.get("text_after", text_after_enrich)

        before_body, before_images = _split_body_and_image_lines(text_before)
        after_body, after_images = _split_body_and_image_lines(text_after)
        image_parts = []
        captions = chunk.get("image_captions")
        if captions:
            image_parts.append("Image Captions:\n" + _json_pretty(captions))
        if after_images:
            image_parts.append("Image Lines in Final Text:\n" + after_images)
        elif before_images:
            image_parts.append("Image Lines:\n" + before_images)
        images_block = "\n\n".join(image_parts) if image_parts else "_该 chunk 无图片相关内容_"

        return (
            info,
            before_body or text_before,
            _split_body_and_image_lines(text_after_refine)[0] or text_after_refine,
            _split_body_and_image_lines(text_after_enrich)[0] or text_after_enrich,
            after_body or text_after,
            images_block,
        )

    def render_embed_chunk(self, choice_value: Optional[str] = None) -> Tuple[str, str, str, pd.DataFrame]:
        if not self._embed_chunks:
            return "_无 Embed 记录_", "", "", pd.DataFrame(columns=["index", "weight"])
        index = self._resolve_index(choice_value, len(self._embed_chunks))
        chunk = self._embed_chunks[index]
        info = (
            f"**Chunk #{chunk.get('chunk_index', index) + 1}** · "
            f"`{chunk.get('parent_id', '')}` · {chunk.get('char_len', 0):,} chars · "
            f"Sparse 非零项 {chunk.get('sparse_nonzero_terms', 0)}"
        )
        input_text = chunk.get("input_text", "")
        body, image_lines = _split_body_and_image_lines(input_text)
        if image_lines:
            input_text = body + "\n\n--- 图片占位符（已从正文分离显示） ---\n" + image_lines
        tokens = chunk.get("tokenized_terms", [])
        tokens_text = ", ".join(tokens)
        pairs = chunk.get("sparse_pairs", [])
        sparse_df = pd.DataFrame(pairs) if pairs else pd.DataFrame(columns=["index", "weight"])
        return info, input_text, tokens_text, sparse_df

    def _split_parent_dropdown_update(self):
        from gradio import update as gr_update

        options = self._choice_options(
            self._split_parents,
            lambda i, p: (
                f"#{p.get('parent_index', i) + 1} · {p.get('parent_id', '')} · "
                f"{p.get('char_len', 0)} chars"
            ),
        )
        value = options[0][1] if options else None
        return gr_update(choices=options, value=value)

    def _split_child_dropdown_update(self):
        from gradio import update as gr_update

        options = self._choice_options(
            self._split_children,
            lambda i, c: (
                f"#{c.get('chunk_index', i) + 1} · {c.get('parent_id', '')} · "
                f"{c.get('char_len', 0)} chars · imgs={len(c.get('image_refs') or [])}"
            ),
        )
        value = options[0][1] if options else None
        return gr_update(choices=options, value=value)

    def _transform_dropdown_update(self):
        from gradio import update as gr_update

        options = self._choice_options(
            self._transform_chunks,
            lambda i, c: (
                f"#{c.get('chunk_index', i) + 1} · {c.get('parent_id', '')} · "
                f"{c.get('char_len', 0)} chars"
            ),
        )
        value = options[0][1] if options else None
        return gr_update(choices=options, value=value)

    def _embed_dropdown_update(self):
        from gradio import update as gr_update

        options = self._choice_options(
            self._embed_chunks,
            lambda i, c: (
                f"#{c.get('chunk_index', i) + 1} · {c.get('parent_id', '')} · "
                f"sparse={c.get('sparse_nonzero_terms', 0)}"
            ),
        )
        value = options[0][1] if options else None
        return gr_update(choices=options, value=value)

    def empty_state(self) -> Tuple[Any, ...]:
        blank = (
            "",
            pd.DataFrame(columns=["#", "Parent ID", "Chars"]),
            gr_update_index_dropdown(),
            "",
            "",
            "",
            "",
            pd.DataFrame(columns=["#", "Parent ID", "Chars", "Image Refs"]),
            gr_update_index_dropdown(),
            "",
            "",
            "",
            "",
            "",
            "",
            pd.DataFrame(columns=["#", "Parent ID", "Chars", "Refined", "Enriched", "Title"]),
            gr_update_index_dropdown(),
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            pd.DataFrame(columns=["#", "Parent ID", "Chars", "Sparse Terms", "Token Count"]),
            gr_update_index_dropdown(),
            "",
            "",
            "",
            pd.DataFrame(columns=["index", "weight"]),
        )
        return (
            gr_update_dropdown([]),
            "_暂无入库追踪记录。请先上传文档完成一次入库。_",
            "",
            _empty_timings_df(),
            "",
            "",
            "",
            *blank,
            "",
        )

    def load_trace_view(self, choice: Optional[str]) -> Tuple[Any, ...]:
        from gradio import update as gr_update

        trace = self._get_trace(choice)
        if trace is None:
            if not self._traces:
                return self.empty_state()
            choice = self.trace_choices()[0]
            trace = self._traces[0]

        self._cache_trace_chunks(trace)
        stages = self._stages_by_name(trace)
        load_d = stages.get("load", {}).get("data", {})
        split_d = stages.get("split", {}).get("data", {})
        transform_d = stages.get("transform", {}).get("data", {})
        embed_d = stages.get("embed", {}).get("data", {})
        upsert_d = stages.get("upsert", {}).get("data", {})

        meta = trace.get("metadata", {})
        source_path = meta.get("source_path", "—")
        total_ms = trace.get("elapsed_ms")
        total_label = f"{total_ms:.0f} ms" if isinstance(total_ms, (int, float)) else "—"
        status = "成功" if trace.get("success") else "失败"

        overview = (
            f"### 📊 Pipeline Overview\n\n"
            f"- **文件**: `{source_path}`\n"
            f"- **状态**: {status}\n"
            f"- **Trace ID**: `{trace.get('trace_id', '—')}`\n"
            f"- **总耗时**: {total_label}\n"
            f"- **文档长度**: {load_d.get('text_length', 0):,} chars\n"
            f"- **父块 / 子块**: {split_d.get('parent_count', 0)} / {split_d.get('child_count', 0)}\n"
            f"- **图片数**: {load_d.get('image_count', 0)}\n"
            f"- **向量维度**: {embed_d.get('dense_dimension', 0)}\n"
            f"- **入库向量**: {upsert_d.get('vector_count', upsert_d.get('child_count', 0))}\n"
        )
        if trace.get("error"):
            overview += f"\n- **错误**: {trace['error']}\n"

        diagnostics = self._format_diagnostics(stages, load_d, split_d, transform_d, embed_d, upsert_d)
        timings_df = self._format_timings_df(stages)
        load_summary, load_body, load_images = self._format_load_stage(load_d, stages.get("load", {}))
        split_summary = self._format_split_summary(split_d)
        transform_summary = self._format_transform_summary(transform_d)
        embed_summary, embed_df = self._format_embed_summary(embed_d, stages.get("embed", {}))
        upsert_summary = self._format_upsert_stage(upsert_d)

        p_info, p_meta, p_text, p_imgs = self.render_split_parent("0")
        c_info, c_meta, c_text, c_img_lines, c_img_meta = self.render_split_child("0")
        t_info, t_before, t_ref, t_enr, t_final, t_imgs = self.render_transform_chunk("0")
        e_info, e_text, e_tokens, e_sparse_df = self.render_embed_chunk("0")

        return (
            gr_update(choices=self.trace_choices(), value=choice),
            overview,
            diagnostics,
            timings_df,
            load_summary,
            load_body,
            load_images,
            split_summary,
            self._split_parent_index_df(),
            self._split_parent_dropdown_update(),
            p_info,
            p_meta,
            p_text,
            p_imgs,
            self._split_child_index_df(),
            self._split_child_dropdown_update(),
            c_info,
            c_meta,
            c_text,
            c_img_lines,
            c_img_meta,
            transform_summary,
            self._transform_index_df(),
            self._transform_dropdown_update(),
            t_info,
            t_before,
            t_ref,
            t_enr,
            t_final,
            t_imgs,
            embed_summary,
            embed_df,
            self._embed_dropdown_update(),
            e_info,
            e_text,
            e_tokens,
            e_sparse_df,
            upsert_summary,
        )

    def _format_timings_df(self, stages: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        rows = []
        for stage_name in MAIN_STAGES:
            stage = stages.get(stage_name)
            if not stage:
                continue
            elapsed = stage.get("elapsed_ms")
            rows.append(
                {
                    "Stage": STAGE_LABELS.get(stage_name, stage_name),
                    "Elapsed (ms)": round(elapsed, 2) if isinstance(elapsed, (int, float)) else "—",
                }
            )
        for stage_name, stage in stages.items():
            if stage_name in MAIN_STAGES:
                continue
            elapsed = stage.get("elapsed_ms")
            rows.append(
                {
                    "Stage": STAGE_LABELS.get(stage_name, stage_name),
                    "Elapsed (ms)": round(elapsed, 2) if isinstance(elapsed, (int, float)) else "—",
                }
            )
        return pd.DataFrame(rows) if rows else _empty_timings_df()

    @staticmethod
    def _format_diagnostics(
        stages: Dict[str, Dict[str, Any]],
        load_d: Dict[str, Any],
        split_d: Dict[str, Any],
        transform_d: Dict[str, Any],
        embed_d: Dict[str, Any],
        upsert_d: Dict[str, Any],
    ) -> str:
        lines: List[str] = []
        missing = [name for name in MAIN_STAGES if name not in stages]
        if missing:
            labels = ", ".join(STAGE_LABELS.get(name, name) for name in missing)
            lines.append(f"⚠️ **Pipeline 不完整，缺少阶段**: {labels}")

        if "load" in stages and load_d.get("text_length", 0) == 0:
            lines.append("⚠️ **Load 阶段文本为空**，文档可能是纯图片或格式不受支持。")

        if "split" in stages and split_d.get("child_count", 0) == 0:
            lines.append("⚠️ **Split 阶段未产生子块**，文档可能过短。")

        if "transform" in stages:
            refined_total = transform_d.get("refined_by_llm", 0) + transform_d.get("refined_by_rule", 0)
            if refined_total == 0:
                lines.append("ℹ️ **Transform**: 没有 chunk 被精炼，可能 LLM 精炼未启用。")

        if "embed" in stages and embed_d.get("dense_vector_count", 0) == 0:
            lines.append("⚠️ **Embed 阶段未生成向量**，Embedding 可能失败。")

        if "upsert" in stages and upsert_d.get("vector_count", upsert_d.get("child_count", 0)) == 0:
            lines.append("⚠️ **Upsert 未写入向量**，入库可能失败。")

        return "\n\n".join(lines)

    @staticmethod
    def _format_load_stage(load_d: Dict[str, Any], stage: Dict[str, Any]) -> Tuple[str, str, str]:
        elapsed = stage.get("elapsed_ms")
        elapsed_label = f"{elapsed:.1f} ms" if isinstance(elapsed, (int, float)) else "—"
        images = load_d.get("images", [])
        summary = (
            f"**Doc Type**: `{load_d.get('doc_type', '—')}`  \n"
            f"**Title**: {load_d.get('title') or '—'}  \n"
            f"**Doc Hash**: `{load_d.get('doc_hash', '—')}`  \n"
            f"**Text Length**: {load_d.get('text_length', 0):,} chars  \n"
            f"**Images**: {load_d.get('image_count', 0)}  \n"
            f"**Markdown Path**: `{load_d.get('md_path', '—')}`  \n"
            f"**Elapsed**: {elapsed_label}  \n\n"
            f"ℹ️ 正文与图片占位符/元数据已分开展示。"
        )

        full_text = load_d.get("text_full") or load_d.get("text_preview") or ""
        body, image_lines = _split_body_and_image_lines(full_text)
        if images:
            images_block = _json_pretty(images)
        elif image_lines:
            images_block = image_lines
        else:
            images_block = "_无图片_"

        if not body and not full_text:
            body = "_该 trace 未记录完整文本。请重新入库。_"
        elif not body:
            body = full_text
        return summary, body, images_block

    @staticmethod
    def _format_split_summary(split_d: Dict[str, Any]) -> str:
        return (
            f"**Parent Chunks**: {split_d.get('parent_count', 0)}  \n"
            f"**Child Chunks**: {split_d.get('child_count', 0)}  \n"
            f"**Avg Size**: {split_d.get('avg_chunk_size', 0)} chars  \n\n"
            f"ℹ️ 子块 metadata 不再重复整份文档的 `images` 列表；"
            f"仅在该 chunk 正文含 `[IMAGE:...]` 时，在「图片引用」区展示对应项。"
        )

    @staticmethod
    def _format_transform_summary(transform_d: Dict[str, Any]) -> str:
        return (
            f"**Refined (LLM / Rule)**: "
            f"{transform_d.get('refined_by_llm', 0)} / {transform_d.get('refined_by_rule', 0)}  \n"
            f"**Enriched (LLM / Rule)**: "
            f"{transform_d.get('enriched_by_llm', 0)} / {transform_d.get('enriched_by_rule', 0)}  \n"
            f"**Captioned Chunks**: {transform_d.get('captioned_chunks', 0)}  \n\n"
            f"ℹ️ 使用下方下拉框逐块查看；正文与图片分栏展示。"
        )

    @staticmethod
    def _format_embed_summary(
        embed_d: Dict[str, Any], stage: Dict[str, Any]
    ) -> Tuple[str, pd.DataFrame]:
        elapsed = stage.get("elapsed_ms")
        elapsed_label = f"{elapsed:.1f} ms" if isinstance(elapsed, (int, float)) else "—"
        summary = (
            f"**Method**: `{embed_d.get('method', 'hybrid')}`  \n"
            f"**Dense Model**: `{embed_d.get('dense_model', config.DENSE_MODEL)}`  \n"
            f"**Sparse Model**: `{embed_d.get('sparse_model', config.SPARSE_MODEL)}`  \n"
            f"**Dense Vectors**: {embed_d.get('dense_vector_count', 0)}  \n"
            f"**Dimension**: {embed_d.get('dense_dimension', 0)}  \n"
            f"**Sparse Docs**: {embed_d.get('sparse_doc_count', 0)}  \n"
            f"**Elapsed**: {elapsed_label}  \n\n"
            f"ℹ️ 选择 chunk 查看编码前文本、BM25 词元列表、稀疏向量 index/weight。"
        )
        rows = []
        for chunk in embed_d.get("chunks", []):
            rows.append(
                {
                    "#": chunk.get("chunk_index", 0) + 1,
                    "Parent ID": chunk.get("parent_id", ""),
                    "Chars": chunk.get("char_len", 0),
                    "Sparse Terms": chunk.get("sparse_nonzero_terms", "—"),
                    "Token Count": len(chunk.get("tokenized_terms", [])),
                }
            )
        embed_df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["#", "Parent ID", "Chars", "Sparse Terms", "Token Count"]
        )
        return summary, embed_df

    @staticmethod
    def _format_upsert_stage(upsert_d: Dict[str, Any]) -> str:
        dense = upsert_d.get("dense_store", {})
        parent = upsert_d.get("parent_store", {})
        images = upsert_d.get("image_store", {})
        vector_ids = upsert_d.get("vector_ids", [])
        lines = [
            f"**Vectors**: {upsert_d.get('vector_count', upsert_d.get('child_count', 0))}  ",
            f"**Dense Store**: `{dense.get('backend', 'Qdrant')}` / `{dense.get('collection', '—')}`  ",
            f"**Dense Path**: `{dense.get('path', config.QDRANT_DB_PATH)}`  ",
            f"**Parent Store**: `{parent.get('backend', 'JSON')}` × {parent.get('count', 0)}  ",
            f"**Parent Path**: `{parent.get('path', config.PARENT_STORE_PATH)}`  ",
            f"**Images Indexed**: {images.get('count', 0)}  ",
            f"**Image Path**: `{images.get('path', config.INGESTION_IMAGES_DIR)}`",
        ]
        if vector_ids:
            lines.append("\n**Vector IDs（全部）**\n```json\n" + _json_pretty(vector_ids) + "\n```")
        return "\n".join(lines)

    def refresh_and_load(self, limit: int = 50) -> Tuple[Any, ...]:
        self.refresh_traces(limit=limit)
        if not self._traces:
            return self.empty_state()
        return self.load_trace_view(self.trace_choices()[0])

    @staticmethod
    def trace_file_status() -> str:
        trace_file = Path(config.INGESTION_TRACE_FILE)
        if not trace_file.exists():
            return f"追踪文件尚未创建：`{config.INGESTION_TRACE_FILE}`"
        line_count = sum(1 for line in trace_file.read_text(encoding="utf-8").splitlines() if line.strip())
        return f"追踪文件：`{config.INGESTION_TRACE_FILE}`（共 {line_count} 条记录）"


def gr_update_dropdown(choices: List[str]):
    from gradio import update as gr_update

    return gr_update(choices=choices, value=choices[0] if choices else None)


def gr_update_index_dropdown():
    from gradio import update as gr_update

    return gr_update(choices=[], value=None)
