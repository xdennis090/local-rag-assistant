from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
UPLOADED_DOCS_DIR = DATA_DIR / "uploaded_docs"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADED_DOCS_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = "local_rag_documents"

DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large"
AVAILABLE_CHAT_MODELS = [
    "llama3.2",
    "llama3.1:8b",
    "qwen2.5:7b",
    "deepseek-r1:1.5b",
]
DEFAULT_CHAT_MODEL = "llama3.2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 4
DEFAULT_CONTEXT_SIZE = 4096
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_HISTORY = 6