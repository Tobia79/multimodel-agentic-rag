import gradio as gr
from core.chat_interface import ChatInterface
from core.document_manager import DocumentManager
from core.evaluation import empty_per_source_dataframe, empty_results_preview_dataframe
from core.evaluation_interface import EvaluationInterface
from core.rag_system import RAGSystem
from ui.ingestion_trace_interface import IngestionTraceInterface
import os

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")

def create_gradio_ui():
    rag_system = RAGSystem()
    rag_system.initialize()
    
    doc_manager = DocumentManager(rag_system)
    chat_interface = ChatInterface(rag_system)
    evaluation_interface = EvaluationInterface(rag_system)
    ingestion_trace_interface = IngestionTraceInterface(doc_manager)
    
    def format_file_list(*, full: bool = True):
        """Document table; full=True queries Qdrant for accurate child-chunk counts."""
        return doc_manager.format_document_table(fast=not full)
    
    def refresh_document_choices(full: bool = False):
        choices = doc_manager.document_choices(fast=not full)
        return gr.update(choices=choices, value=choices[0] if choices else None), doc_manager.format_document_table(fast=not full)

    def initial_page_load():
        choices = doc_manager.document_choices(fast=True)
        return (
            EvaluationInterface.kb_status(),
            gr.update(choices=choices, value=choices[0] if choices else None),
            format_file_list(),
        )
    
    def upload_handler(files, progress=gr.Progress()):
        if not files:
            choices = doc_manager.document_choices(fast=True)
            return (
                None,
                gr.update(choices=choices, value=choices[0] if choices else None),
                format_file_list(),
                EvaluationInterface.kb_status(),
            )
            
        summary = doc_manager.sync_documents(
            files,
            progress_callback=lambda p, desc: progress(p, desc=desc),
        )

        gr.Info(
            f"✅ 新增：{summary.added} | 更新：{summary.updated} | "
            f"跳过：{summary.skipped} | 失败：{summary.failed}"
        )
        choices = doc_manager.document_choices(fast=False)
        return (
            None,
            gr.update(choices=choices, value=choices[0] if choices else None),
            doc_manager.format_document_table(fast=False),
            EvaluationInterface.kb_status(),
        )
    
    def clear_handler():
        doc_manager.clear_all()
        gr.Info("🗑️ 已清空所有文档")
        return gr.update(choices=[], value=None), format_file_list(), EvaluationInterface.kb_status()

    def delete_document_handler(document_name):
        if not document_name:
            gr.Warning("请先选择要删除的文档")
            return format_file_list(), gr.update(), EvaluationInterface.kb_status()
        result = doc_manager.delete_document(document_name)
        if result.success:
            gr.Info(
                f"已删除 {document_name}：向量 {result.chunks_deleted}，"
                f"父块 {result.parents_deleted}，图片目录 {result.images_deleted}"
            )
        else:
            gr.Warning(f"删除 {document_name} 部分失败：{'；'.join(result.errors)}")
        choices = doc_manager.document_choices(fast=False)
        return (
            doc_manager.format_document_table(fast=False),
            gr.update(choices=choices, value=choices[0] if choices else None),
            EvaluationInterface.kb_status(),
        )
    
    def chat_handler(msg, hist, force_rag=False):
        for chunk in chat_interface.chat(msg, hist, force_rag=force_rag):
            yield chunk
    
    def clear_chat_handler():
        chat_interface.clear_session()

    def evaluation_handler(sample_size, mode, dataset_file, progress=gr.Progress()):
        for update in evaluation_interface.run(
            sample_size=int(sample_size),
            mode=mode,
            progress=progress,
            dataset_csv=dataset_file,
        ):
            yield update

    def refresh_kb_handler():
        return EvaluationInterface.kb_status()

    def preview_eval_config(sample_size, mode, dataset_file):
        return EvaluationInterface.config_summary(int(sample_size), mode, dataset_csv=dataset_file)

    def toggle_dataset_upload(mode):
        return gr.update(visible=mode == "score_only")

    def refresh_ingestion_traces():
        return ingestion_trace_interface.refresh_and_load()

    def load_ingestion_trace(choice):
        ingestion_trace_interface.refresh_traces()
        return ingestion_trace_interface.load_trace_view(choice)

    def on_split_parent_select(choice_value):
        return ingestion_trace_interface.render_split_parent(choice_value)

    def on_split_child_select(choice_value):
        return ingestion_trace_interface.render_split_child(choice_value)

    def on_transform_chunk_select(choice_value):
        return ingestion_trace_interface.render_transform_chunk(choice_value)

    def on_embed_chunk_select(choice_value):
        return ingestion_trace_interface.render_embed_chunk(choice_value)
    
    with gr.Blocks(title="Agentic RAG 助手") as demo:
        with gr.Tab("文档管理", elem_id="doc-management-tab"):
            gr.Markdown("## 添加文档")
            gr.Markdown(
                "上传 PDF、Markdown 或 Word（.docx/.doc）文件。"
                "已处理且内容未变的文件将自动跳过（基于 SHA256）。"
            )
            
            files_input = gr.File(
                label="将 PDF 或 Markdown 文件拖放到此处",
                file_count="multiple",
                type="filepath",
                height=200,
                show_label=False
            )
            
            add_btn = gr.Button("添加文档", variant="primary", size="md")
            
            gr.Markdown("## 知识库中的文档")
            file_list = gr.Markdown(value=format_file_list())

            with gr.Row():
                delete_dropdown = gr.Dropdown(
                    label="选择要删除的文档",
                    choices=doc_manager.document_choices(fast=True),
                    interactive=True,
                    scale=4,
                )
                delete_btn = gr.Button("删除所选文档", variant="stop", size="md", scale=1)
            
            with gr.Row():
                refresh_btn = gr.Button("刷新", size="md")
                clear_btn = gr.Button("清空全部", variant="stop", size="md")

        with gr.Tab("入库追踪", elem_id="ingestion-traces-tab"):
            gr.Markdown("## 🔬 Ingestion Traces")
            gr.Markdown(
                "浏览每次文档入库的 Pipeline 阶段详情：解析、分块、清洗丰富、写入向量库。"
                "参考 MODULAR-RAG 的 Ingestion Traces 面板设计。"
            )
            trace_file_status = gr.Markdown(value=IngestionTraceInterface.trace_file_status())

            with gr.Row():
                trace_dropdown = gr.Dropdown(
                    label="选择追踪记录",
                    choices=[],
                    interactive=True,
                    scale=4,
                )
                refresh_traces_btn = gr.Button("刷新追踪", size="md", scale=1)

            overview_md = gr.Markdown(value="_加载中…_")
            diagnostics_md = gr.Markdown()
            timing_df = gr.Dataframe(
                label="阶段耗时",
                interactive=False,
                wrap=True,
            )

            with gr.Tabs():
                with gr.Tab("📄 Load"):
                    load_summary = gr.Markdown()
                    load_preview = gr.Textbox(
                        label="文档正文（不含图片占位符）",
                        lines=18,
                        max_lines=80,
                        interactive=False,
                    )
                    load_images_tb = gr.Textbox(
                        label="图片元数据 / 占位符",
                        lines=10,
                        max_lines=40,
                        interactive=False,
                    )
                with gr.Tab("✂️ Split"):
                    split_summary = gr.Markdown()
                    with gr.Tabs():
                        with gr.Tab("父块"):
                            split_parent_df = gr.Dataframe(
                                label="父块索引",
                                interactive=False,
                                wrap=True,
                            )
                            split_parent_select = gr.Dropdown(
                                label="选择父块",
                                choices=[],
                                interactive=True,
                            )
                            split_parent_info = gr.Markdown()
                            split_parent_metadata = gr.Textbox(
                                label="Metadata（不含 images）",
                                lines=8,
                                max_lines=40,
                                interactive=False,
                            )
                            split_parent_text = gr.Textbox(
                                label="正文",
                                lines=14,
                                max_lines=80,
                                interactive=False,
                            )
                            split_parent_images = gr.Textbox(
                                label="图片占位符行",
                                lines=6,
                                max_lines=30,
                                interactive=False,
                            )
                        with gr.Tab("子块"):
                            split_child_df = gr.Dataframe(
                                label="子块索引",
                                interactive=False,
                                wrap=True,
                            )
                            split_child_select = gr.Dropdown(
                                label="选择子块",
                                choices=[],
                                interactive=True,
                            )
                            split_child_info = gr.Markdown()
                            split_child_metadata = gr.Textbox(
                                label="Metadata（不含整份文档 images）",
                                lines=8,
                                max_lines=40,
                                interactive=False,
                            )
                            split_child_text = gr.Textbox(
                                label="正文（text_body，不含 [IMAGE:...] 行）",
                                lines=14,
                                max_lines=80,
                                interactive=False,
                            )
                            split_child_image_lines = gr.Textbox(
                                label="图片占位符行",
                                lines=6,
                                max_lines=30,
                                interactive=False,
                            )
                            split_child_images_meta = gr.Textbox(
                                label="该子块引用的图片元数据",
                                lines=8,
                                max_lines=40,
                                interactive=False,
                            )
                with gr.Tab("🔄 Transform"):
                    transform_summary = gr.Markdown()
                    transform_chunk_df = gr.Dataframe(
                        label="Chunk 索引",
                        interactive=False,
                        wrap=True,
                    )
                    transform_chunk_select = gr.Dropdown(
                        label="选择 Chunk",
                        choices=[],
                        interactive=True,
                    )
                    transform_chunk_info = gr.Markdown()
                    with gr.Row():
                        transform_before = gr.Textbox(
                            label="Refine 前正文",
                            lines=10,
                            max_lines=60,
                            interactive=False,
                        )
                        transform_after_refine = gr.Textbox(
                            label="Refine 后正文",
                            lines=10,
                            max_lines=60,
                            interactive=False,
                        )
                    with gr.Row():
                        transform_after_enrich = gr.Textbox(
                            label="Enrich 后正文",
                            lines=10,
                            max_lines=60,
                            interactive=False,
                        )
                        transform_after_final = gr.Textbox(
                            label="最终正文",
                            lines=10,
                            max_lines=60,
                            interactive=False,
                        )
                    transform_images = gr.Textbox(
                        label="图片 / Caption",
                        lines=8,
                        max_lines=40,
                        interactive=False,
                    )
                with gr.Tab("🔢 Embed"):
                    embed_summary = gr.Markdown()
                    embed_df = gr.Dataframe(
                        label="编码概览（全部 chunk）",
                        interactive=False,
                        wrap=True,
                    )
                    embed_chunk_select = gr.Dropdown(
                        label="选择 Chunk",
                        choices=[],
                        interactive=True,
                    )
                    embed_chunk_info = gr.Markdown()
                    embed_input_text = gr.Textbox(
                        label="编码前文本",
                        lines=12,
                        max_lines=60,
                        interactive=False,
                    )
                    embed_tokens = gr.Textbox(
                        label="BM25 词元列表（编码前）",
                        lines=8,
                        max_lines=40,
                        interactive=False,
                    )
                    embed_sparse_df = gr.Dataframe(
                        label="稀疏向量 (index, weight)",
                        interactive=False,
                        wrap=True,
                    )
                with gr.Tab("💾 Upsert"):
                    upsert_summary = gr.Markdown()
        
        with gr.Tab("对话"):
            chatbot = gr.Chatbot(
                height=720, 
                placeholder="<strong>问我任何问题！</strong><br><em>我会搜索、推理并行动，为你提供最佳答案 :)</em>",
                show_label=False,
                avatar_images=(None, os.path.join(ASSETS_DIR, "chatbot_avatar.png")),
                layout="bubble"
            )
            chatbot.clear(clear_chat_handler)

            force_rag_cb = gr.Checkbox(
                label="强制查知识库",
                value=False,
                info="勾选后跳过直答路由，始终检索已上传文档",
            )

            gr.ChatInterface(
                fn=chat_handler,
                chatbot=chatbot,
                additional_inputs=[force_rag_cb],
                textbox=gr.Textbox(
                    placeholder="输入消息…",
                    show_label=False,
                    lines=1,
                    max_lines=5,
                ),
            )

        with gr.Tab("评估", elem_id="evaluation-tab"):
            kb_status = gr.Markdown(value=EvaluationInterface.kb_status())

            gr.Markdown(
                "## RAGAS 评估\n"
                "使用 30 题基准测试集评估当前知识库上的 Agentic RAG 表现。"
            )

            with gr.Row():
                refresh_kb_btn = gr.Button("刷新知识库状态", size="sm")
                sample_input = gr.Dropdown(
                    choices=[
                        ("5 题（快速）", 5),
                        ("10 题", 10),
                        ("30 题（完整）", 30),
                    ],
                    value=5,
                    label="题量",
                    scale=1,
                )
                mode_input = gr.Radio(
                    choices=[
                        ("完整评估（查询 + 打分）", "full"),
                        ("仅查询（保存 dataset）", "query_only"),
                        ("仅打分（读取已有 CSV）", "score_only"),
                    ],
                    value="full",
                    label="评估模式",
                    scale=2,
                )

            with gr.Row():
                run_eval_btn = gr.Button("开始评估", variant="primary", size="md", scale=1)

            dataset_csv_input = gr.File(
                label="上传 dataset CSV（仅打分模式，可选）",
                file_types=[".csv"],
                type="filepath",
                visible=False,
            )

            eval_config_preview = gr.Markdown(
                value=EvaluationInterface.config_summary(5, "full")
            )

            eval_status = gr.Markdown(
                value="_等待评估。配置见上方，完成后结果将显示在下方。_"
            )

            results_accordion = gr.Accordion("评估结果", open=False, visible=False)
            with results_accordion:
                scoring_live_log = gr.Markdown(value="## Live Scores\n\n_Waiting to start…_")
                metric_summary = gr.Markdown(value="_Mean scores will appear here after evaluation._")
                per_source_table = gr.Dataframe(
                    value=empty_per_source_dataframe(),
                    label="Mean scores per source",
                    interactive=False,
                    wrap=True,
                )
                results_table = gr.Dataframe(
                    value=empty_results_preview_dataframe(),
                    label="Per-question results",
                    interactive=False,
                    wrap=True,
                )
                radar_chart = gr.Image(
                    label="Overall RAGAS Profile",
                    type="filepath",
                    visible=False,
                    height=400,
                )

            downloads_row = gr.Row(visible=False)
            with downloads_row:
                dataset_download = gr.File(label="下载查询 dataset CSV")
                results_download = gr.File(label="下载完整结果 CSV")

            refresh_kb_btn.click(refresh_kb_handler, None, kb_status)
            sample_input.change(
                preview_eval_config,
                [sample_input, mode_input, dataset_csv_input],
                eval_config_preview,
            )
            mode_input.change(
                preview_eval_config,
                [sample_input, mode_input, dataset_csv_input],
                eval_config_preview,
            )
            mode_input.change(toggle_dataset_upload, mode_input, dataset_csv_input)
            dataset_csv_input.change(
                preview_eval_config,
                [sample_input, mode_input, dataset_csv_input],
                eval_config_preview,
            )

            run_eval_btn.click(
                evaluation_handler,
                inputs=[sample_input, mode_input, dataset_csv_input],
                outputs=[
                    eval_status,
                    scoring_live_log,
                    metric_summary,
                    per_source_table,
                    results_table,
                    radar_chart,
                    dataset_download,
                    results_download,
                    results_accordion,
                    downloads_row,
                    run_eval_btn,
                ],
                show_progress="hidden",
            )

        trace_outputs = [
            trace_dropdown,
            overview_md,
            diagnostics_md,
            timing_df,
            load_summary,
            load_preview,
            load_images_tb,
            split_summary,
            split_parent_df,
            split_parent_select,
            split_parent_info,
            split_parent_metadata,
            split_parent_text,
            split_parent_images,
            split_child_df,
            split_child_select,
            split_child_info,
            split_child_metadata,
            split_child_text,
            split_child_image_lines,
            split_child_images_meta,
            transform_summary,
            transform_chunk_df,
            transform_chunk_select,
            transform_chunk_info,
            transform_before,
            transform_after_refine,
            transform_after_enrich,
            transform_after_final,
            transform_images,
            embed_summary,
            embed_df,
            embed_chunk_select,
            embed_chunk_info,
            embed_input_text,
            embed_tokens,
            embed_sparse_df,
            upsert_summary,
        ]

        split_parent_outputs = [
            split_parent_info,
            split_parent_metadata,
            split_parent_text,
            split_parent_images,
        ]
        split_child_outputs = [
            split_child_info,
            split_child_metadata,
            split_child_text,
            split_child_image_lines,
            split_child_images_meta,
        ]
        transform_chunk_outputs = [
            transform_chunk_info,
            transform_before,
            transform_after_refine,
            transform_after_enrich,
            transform_after_final,
            transform_images,
        ]
        embed_chunk_outputs = [
            embed_chunk_info,
            embed_input_text,
            embed_tokens,
            embed_sparse_df,
        ]

        refresh_traces_btn.click(
            refresh_ingestion_traces,
            None,
            trace_outputs,
        )
        trace_dropdown.change(
            load_ingestion_trace,
            trace_dropdown,
            trace_outputs,
        )
        split_parent_select.change(
            on_split_parent_select,
            split_parent_select,
            split_parent_outputs,
        )
        split_child_select.change(
            on_split_child_select,
            split_child_select,
            split_child_outputs,
        )
        transform_chunk_select.change(
            on_transform_chunk_select,
            transform_chunk_select,
            transform_chunk_outputs,
        )
        embed_chunk_select.change(
            on_embed_chunk_select,
            embed_chunk_select,
            embed_chunk_outputs,
        )

        add_btn.click(
            upload_handler,
            [files_input],
            [files_input, delete_dropdown, file_list, kb_status],
            show_progress="corner",
        )
        refresh_btn.click(lambda: refresh_document_choices(full=True), None, [delete_dropdown, file_list])
        delete_btn.click(
            delete_document_handler,
            [delete_dropdown],
            [file_list, delete_dropdown, kb_status],
        )
        clear_btn.click(clear_handler, None, [delete_dropdown, file_list, kb_status])
        demo.load(initial_page_load, None, [kb_status, delete_dropdown, file_list])
    
    return demo