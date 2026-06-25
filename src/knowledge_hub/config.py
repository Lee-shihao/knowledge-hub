from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KH_", env_file=".env")

    # Server — shared by MCP and HTTP upload
    SERVER_HOST: str = "127.0.0.1"
    MCP_PORT: int = 8765
    # Auth (required for LAN deployment)
    SERVER_AUTH_TOKEN: str | None = None
    SERVER_ALLOWED_IPS: list[str] = []

    # Embedding models (FlagEmbedding HuggingFace IDs)
    EMBED_MODEL: str = "BAAI/bge-m3"
    RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
    EMBED_DEVICE: Literal["auto", "cpu", "cuda"] = "auto"

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_MODE: Literal["embedded", "http"] = "embedded"
    QDRANT_PATH: str = "./storage/qdrant"
    QDRANT_COLLECTION: str = "knowledge_hub"

    # Upload server
    UPLOAD_PORT: int = 8766
    UPLOAD_ENABLED: bool = True

    # Ingestion
    CHUNK_MAX_TOKENS: int = 512
    CHUNK_OVERLAP: float = 0.1
    EMBED_BATCH_SIZE: int = 16
    MAX_FILE_SIZE_MB: int = 200
    WARN_FILE_SIZE_MB: int = 50

    # Query
    HYBRID_CANDIDATE_K: int = 20
    FINAL_TOP_K: int = 5

    # Storage paths
    DATA_DIR: str = "./data"
    STORAGE_DIR: str = "./storage"

