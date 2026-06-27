import os

# --- Directory Configuration ---
_BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MARKDOWN_DIR = os.path.join(_BASE_DIR, "markdown_docs")
PARENT_STORE_PATH = os.path.join(_BASE_DIR, "parent_store")
QDRANT_DB_PATH = os.path.join(_BASE_DIR, "qdrant_db")

# --- Qdrant Configuration ---
CHILD_COLLECTION = "document_child_chunks"
SPARSE_VECTOR_NAME = "sparse"

# --- Model Configuration ---
# Local sentence-transformers model (768-dim). Re-index qdrant_db after switching embeddings.
DENSE_MODEL = os.path.join(_BASE_DIR, "all-mpnet-base-v2")
# Local BM25 stopwords (offline). Download once: hf download Qdrant/bm25 --local-dir Qdrant-bm25
SPARSE_MODEL = "Qdrant/bm25"
SPARSE_MODEL_PATH = os.path.join(_BASE_DIR, "Qdrant-bm25")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")  # ollama | deepseek
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:4b-instruct-2507-q4_K_M")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0"))
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# --- Retrieval Configuration (MODULAR-RAG style: dense + sparse -> RRF -> rerank) ---
DENSE_TOP_K = max(1, int(os.environ.get("DENSE_TOP_K", "20")))
SPARSE_TOP_K = max(1, int(os.environ.get("SPARSE_TOP_K", "20")))
FUSION_TOP_K = max(1, int(os.environ.get("FUSION_TOP_K", "10")))
RRF_K = max(1, int(os.environ.get("RRF_K", "60")))
# Final k for agent tools and evaluation when not overridden by the caller.
DEFAULT_RETRIEVAL_K = FUSION_TOP_K

# --- Rerank Configuration ---
RERANK_ENABLED = os.environ.get("RERANK_ENABLED", "true").lower() == "true"
RERANK_PROVIDER = os.environ.get("RERANK_PROVIDER", "cross_encoder").strip().lower()
RERANK_MODEL = os.environ.get(
    "RERANK_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
RERANK_CANDIDATE_MULTIPLIER = max(
    1,
    int(os.environ.get("RERANK_CANDIDATE_MULTIPLIER", "2")),
)
RERANK_TIMEOUT_SEC = float(os.environ.get("RERANK_TIMEOUT_SEC", "30"))

# --- Retrieval Confidence (layered: rerank -> optional LLM -> agent routing) ---
CONFIDENCE_ENABLED = os.environ.get("CONFIDENCE_ENABLED", "true").lower() == "true"
CONFIDENCE_SECONDARY_RETRIEVAL = os.environ.get("CONFIDENCE_SECONDARY_RETRIEVAL", "true").lower() == "true"
CONFIDENCE_SECONDARY_MULTIPLIER = max(
    2,
    int(os.environ.get("CONFIDENCE_SECONDARY_MULTIPLIER", "2")),
)
CONFIDENCE_SECONDARY_DENSE_TOP_K = max(
    1,
    int(os.environ.get("CONFIDENCE_SECONDARY_DENSE_TOP_K", "30")),
)
CONFIDENCE_SECONDARY_SPARSE_TOP_K = max(
    1,
    int(os.environ.get("CONFIDENCE_SECONDARY_SPARSE_TOP_K", "30")),
)
CONFIDENCE_RERANK_LOW_THRESHOLD = float(os.environ.get("CONFIDENCE_RERANK_LOW_THRESHOLD", "-2.0"))
# Rerank top score range mapped linearly to preliminary confidence 0-10 (continuous, no gaps).
CONFIDENCE_RERANK_SCORE_MIN = float(os.environ.get("CONFIDENCE_RERANK_SCORE_MIN", "-5.0"))
CONFIDENCE_RERANK_SCORE_MAX = float(os.environ.get("CONFIDENCE_RERANK_SCORE_MAX", "10.0"))
# Preliminary (粗估) in [GRAY_LOW, GRAY_HIGH] triggers LLM fine evaluation.
CONFIDENCE_RERANK_GRAY_LOW = float(os.environ.get("CONFIDENCE_RERANK_GRAY_LOW", "2"))
CONFIDENCE_RERANK_GRAY_HIGH = float(os.environ.get("CONFIDENCE_RERANK_GRAY_HIGH", "8"))
CONFIDENCE_LLM_ENABLED = os.environ.get("CONFIDENCE_LLM_ENABLED", "true").lower() == "true"
CONFIDENCE_LLM_MAX_CONTEXT_CHUNKS = max(
    1,
    int(os.environ.get("CONFIDENCE_LLM_MAX_CONTEXT_CHUNKS", "5")),
)
# Final score tiers after fine eval: high>=HIGH, medium in [LOW, HIGH), low<LOW.
# Medium tier triggers Agent query rewrite + search_child_chunks retry.
CONFIDENCE_HIGH_THRESHOLD = float(os.environ.get("CONFIDENCE_HIGH_THRESHOLD", "7"))
CONFIDENCE_LOW_THRESHOLD = float(os.environ.get("CONFIDENCE_LOW_THRESHOLD", "3"))
CONFIDENCE_AGENT_RETRY_ON_MEDIUM = os.environ.get("CONFIDENCE_AGENT_RETRY_ON_MEDIUM", "true").lower() == "true"
CONFIDENCE_MAX_AGENT_RETRIES = max(0, int(os.environ.get("CONFIDENCE_MAX_AGENT_RETRIES", "1")))
CONFIDENCE_WEB_SEARCH_ON_LOW = os.environ.get("CONFIDENCE_WEB_SEARCH_ON_LOW", "true").lower() == "true"
CONFIDENCE_MAX_WEB_SEARCH_RETRIES = max(0, int(os.environ.get("CONFIDENCE_MAX_WEB_SEARCH_RETRIES", "1")))

# --- Web Search (DuckDuckGo fallback when local KB is insufficient) ---
WEB_SEARCH_ENABLED = os.environ.get("WEB_SEARCH_ENABLED", "true").lower() == "true"
WEB_SEARCH_MAX_RESULTS = max(1, int(os.environ.get("WEB_SEARCH_MAX_RESULTS", "5")))

# --- Query Routing (direct LLM vs RAG) ---
QUERY_ROUTING_ENABLED = os.environ.get("QUERY_ROUTING_ENABLED", "true").lower() == "true"
QUERY_ROUTING_USE_RULES = os.environ.get("QUERY_ROUTING_USE_RULES", "true").lower() == "true"
QUERY_ROUTING_LLM_THRESHOLD = float(os.environ.get("QUERY_ROUTING_LLM_THRESHOLD", "0.7"))
QUERY_ROUTING_RULES_ONLY = os.environ.get("QUERY_ROUTING_RULES_ONLY", "false").lower() == "true"

# --- Agent Configuration ---
MAX_TOOL_CALLS = 8
MAX_ITERATIONS = 10
GRAPH_RECURSION_LIMIT = 50
BASE_TOKEN_THRESHOLD = 2000
TOKEN_GROWTH_FACTOR = 0.9

# --- Text Splitter Configuration ---
CHILD_CHUNK_SIZE = 500
MIN_PARENT_SIZE = 2000
MAX_PARENT_SIZE = 4000
# Semantic chunking: contextual embedding buffer sizes and similarity percentiles.
PARENT_CONTEXT_BUFFER = max(0, int(os.environ.get("PARENT_CONTEXT_BUFFER", "2")))
CHILD_CONTEXT_BUFFER = max(0, int(os.environ.get("CHILD_CONTEXT_BUFFER", "1")))
PARENT_SEMANTIC_PERCENTILE = float(os.environ.get("PARENT_SEMANTIC_PERCENTILE", "90"))
CHILD_SEMANTIC_PERCENTILE = float(os.environ.get("CHILD_SEMANTIC_PERCENTILE", "70"))
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3")
]

# --- Ingestion Pipeline Configuration ---
INGESTION_DB_PATH = os.path.join(_BASE_DIR, "data", "db", "ingestion_history.db")
INGESTION_TRACE_FILE = os.path.join(_BASE_DIR, "logs", "ingestion_traces.jsonl")
INGESTION_IMAGES_DIR = os.path.join(_BASE_DIR, "data", "images")
INGESTION_CONVERTED_DOCX_DIR = os.path.join(_BASE_DIR, "data", "converted_docx")
INGESTION_EXTRACT_IMAGES = os.environ.get("INGESTION_EXTRACT_IMAGES", "true").lower() == "true"
INGESTION_DOC_CONVERT_TO_DOCX = os.environ.get("INGESTION_DOC_CONVERT_TO_DOCX", "true").lower() == "true"
INGESTION_CHUNK_REFINER_USE_LLM = os.environ.get("INGESTION_CHUNK_REFINER_USE_LLM", "false").lower() == "true"
INGESTION_METADATA_ENRICHER_USE_LLM = os.environ.get("INGESTION_METADATA_ENRICHER_USE_LLM", "false").lower() == "true"
INGESTION_PARENT_METADATA_USE_LLM = os.environ.get("INGESTION_PARENT_METADATA_USE_LLM", "true").lower() == "true"
INGESTION_PARENT_MAX_TAGS = max(1, min(10, int(os.environ.get("INGESTION_PARENT_MAX_TAGS", "10"))))
INGESTION_PARENT_LLM_MAX_CHARS = max(500, int(os.environ.get("INGESTION_PARENT_LLM_MAX_CHARS", str(MAX_PARENT_SIZE))))
INGESTION_LLM_MAX_WORKERS = max(1, int(os.environ.get("INGESTION_LLM_MAX_WORKERS", "5")))
INGESTION_VISION_MAX_WORKERS = max(1, int(os.environ.get("INGESTION_VISION_MAX_WORKERS", "3")))
VISION_LLM_ENABLED = os.environ.get("VISION_LLM_ENABLED", "false").lower() == "true"
VISION_LLM_PROVIDER = os.environ.get("VISION_LLM_PROVIDER", "ollama").strip().lower()
VISION_LLM_MODEL = os.environ.get("VISION_LLM_MODEL", "llava")
VISION_LLM_BASE_URL = os.environ.get("VISION_LLM_BASE_URL", "http://127.0.0.1:11434")
VISION_LLM_API_KEY = os.environ.get("VISION_LLM_API_KEY", "")

# --- OCR Configuration (Transform / ImageCaptioner) ---
OCR_ENABLED = os.environ.get("OCR_ENABLED", "false").lower() == "true"
OCR_PROVIDER = os.environ.get("OCR_PROVIDER", "paddle").strip().lower()
OCR_LANG = os.environ.get("OCR_LANG", "ch").strip().lower()
# ocr_only | vlm_only | ocr_then_vlm
IMAGE_UNDERSTANDING_MODE = os.environ.get("IMAGE_UNDERSTANDING_MODE", "ocr_then_vlm").strip().lower()

# --- Scanned PDF (Load stage: page render → OCR → VLM) ---
INGESTION_PDF_SCAN_OCR = os.environ.get("INGESTION_PDF_SCAN_OCR", "true").lower() == "true"
# auto: detect low text layer | always | never
INGESTION_PDF_SCAN_MODE = os.environ.get("INGESTION_PDF_SCAN_MODE", "auto").strip().lower()
INGESTION_PDF_SCAN_DPI = max(72, min(400, int(os.environ.get("INGESTION_PDF_SCAN_DPI", "200"))))
INGESTION_PDF_SCAN_TEXT_THRESHOLD = max(
    1,
    int(os.environ.get("INGESTION_PDF_SCAN_TEXT_THRESHOLD", "50")),
)

# --- Langfuse Observability ---
LANGFUSE_ENABLED = os.environ.get("LANGFUSE_ENABLED", "false").lower() == "true"
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000")
