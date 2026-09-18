from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ─────────────────────────────────────
    # Gemini
    # ─────────────────────────────────────

    google_api_key_1: str = ""
    google_api_key_2: str = ""
    google_api_key_3: str = ""
    google_api_key_4: str = ""

    gemini_model: str = "gemini-2.5-flash"

    llm_temperature: float = 0.3
    llm_max_tokens: int = 1024

    # ─────────────────────────────────────
    # Alibaba / Qwen
    # ─────────────────────────────────────

    alibaba_api_key: str = ""

    alibaba_base_url: str = (
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )

    alibaba_models: str = (
        "qwen-plus-character,qwen-flash-character"
    )

    # Maximum time one LLM/model can wait
    # before switching to another model/key.
    llm_timeout_seconds: int = 8

    # ─────────────────────────────────────
    # App
    # ─────────────────────────────────────

    # development = local
    # staging     = testing
    # production  = live

    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = True

    # ─────────────────────────────────────
    # RAG / Pinecone
    # ─────────────────────────────────────

    pinecone_api_key: str = ""
    pinecone_index_name: str = "sbk-website"
    pinecone_namespace: str = "sbk-website"

    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    embedding_model_name: str = "llama-text-embed-v2"
    embedding_dimension: int = 384
    rag_top_k: int = 3

    # ─────────────────────────────────────
    # Memory
    # ─────────────────────────────────────

    memory_backend: str = "redis"

    max_summaries: int = 5

    # ─────────────────────────────────────
    # Redis
    # ─────────────────────────────────────

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""

    # ─────────────────────────────────────
    # CORS
    # ─────────────────────────────────────

    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ]


# Singleton instance used across the app
settings = Settings()