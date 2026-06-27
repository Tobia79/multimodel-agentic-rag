# Multimodel Agentic RAG

> 个人学习项目，基于 [agentic-rag-for-dummies](https://github.com/GiovanniPasq/agentic-rag-for-dummies) 扩展实践，用于探索多模态 Agentic RAG 的搭建与评测，不代表生产级方案。

## 环境配置

```bash
git clone https://github.com/Tobia79/multimodel-agentic-rag.git
cd multimodel-agentic-rag

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
cp project/.env.example project/.env
```

编辑 `project/.env`，至少配置 LLM：

| 场景 | 关键变量 |
|------|----------|
| 本地 Ollama | `LLM_PROVIDER=ollama`，`LLM_MODEL=qwen3:4b-instruct-2507-q4_K_M` |
| DeepSeek API | `LLM_PROVIDER=deepseek`，`DEEPSEEK_API_KEY=你的密钥` |

使用 Ollama 时需先安装并拉取模型：

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

首次运行会自动下载嵌入模型；也可提前下载到项目根目录：

```bash
# 稠密向量
huggingface-cli download sentence-transformers/all-mpnet-base-v2 --local-dir all-mpnet-base-v2
# BM25 稀疏向量
huggingface-cli download Qdrant/bm25 --local-dir Qdrant-bm25
```

将待检索文档放入 `docs/`，启动后通过 Gradio 界面上传并建立索引。

## 启动项目

```bash
python project/app.py
```

浏览器打开 `http://127.0.0.1:7860`，在「对话」页提问，在「文档」页上传 PDF / Markdown 并完成入库。

## 评测项目

评测前需已完成文档入库。测试集默认位于 `notebooks/data/curated_ragas_qa.json`。

**方式一：Gradio 界面**

启动 `project/app.py` 后，进入「评估」标签页，选择样本数量与模式，点击「开始评估」。结果保存在 `data/evaluation/`。

**方式二：命令行**

```bash
cd project
python -m core.evaluation              # 默认抽样 5 题，查询 + RAGAS 打分
python -m core.evaluation --sample 0     # 全部 30 题
python -m core.evaluation --query-only   # 仅生成回答，保存 CSV
python -m core.evaluation --skip-query   # 基于已有 CSV 仅打分
```

**方式三：Notebook**

```bash
jupyter notebook notebooks/evaluation.ipynb
```

按单元格顺序运行即可。指标包括 Answer Accuracy、Context Relevance、Response Groundedness、Context Precision、Context Recall，以及自定义检索指标 hit_rate / mrr。
