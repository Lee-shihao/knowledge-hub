# tests/test_config.py
import os
from knowledge_hub.config import Settings


def test_settings_defaults():
    settings = Settings()
    assert settings.SERVER_HOST == "127.0.0.1"
    assert settings.MCP_PORT == 8765

    assert settings.EMBED_MODEL == "BAAI/bge-m3"
    assert settings.RERANK_MODEL == "BAAI/bge-reranker-v2-m3"
    assert settings.EMBED_DEVICE == "auto"
    assert settings.QDRANT_URL == "http://localhost:6333"
    assert settings.QDRANT_COLLECTION == "knowledge_hub"
    assert settings.CHUNK_MAX_TOKENS == 512
    assert settings.CHUNK_OVERLAP == 0.1
    assert settings.EMBED_BATCH_SIZE == 16
    assert settings.MAX_FILE_SIZE_MB == 200
    assert settings.WARN_FILE_SIZE_MB == 50
    assert settings.HYBRID_CANDIDATE_K == 20
    assert settings.FINAL_TOP_K == 5
    assert settings.SERVER_AUTH_TOKEN is None
    assert settings.SERVER_ALLOWED_IPS == []
    assert settings.DATA_DIR == "./data"
    assert settings.STORAGE_DIR == "./storage"


def test_settings_from_env():
    os.environ["KH_SERVER_HOST"] = "0.0.0.0"
    os.environ["KH_SERVER_AUTH_TOKEN"] = "test-token"
    settings = Settings()
    assert settings.SERVER_HOST == "0.0.0.0"
    assert settings.SERVER_AUTH_TOKEN == "test-token"
    del os.environ["KH_SERVER_HOST"]
    del os.environ["KH_SERVER_AUTH_TOKEN"]
