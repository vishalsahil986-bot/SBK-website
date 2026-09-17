from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8", #Read the .env file using UTF-8 encoding/ UTF-8 supports all characters including special symbols, emojis, other languages
        case_sensitive=False,
    )

    # ── LLM (Gemini) ─────────────────────
    google_api_key_1: str = ""
    google_api_key_2: str = ""
    google_api_key_3: str = ""
    google_api_key_4: str = ""
    gemini_model: str = "gemini-1.5-flash"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 512

    #App
    #"development"	Running locally on your machine 
    #"staging"	Testing before going live
    #"production"	Live app for real users
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = True

        # ── RAG / Pinecone ─────────────────────
    pinecone_api_key: str = ""
    pinecone_index_name: str = "voice-agent"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    embedding_model_name: str = "llama-text-embed-v2"
    embedding_dimension: int = 384
    rag_top_k: int = 3

    #Memory
    memory_backend: str = "redis"      # "dict" | "redis" | "both"
    max_summaries: int = 5

    # Redis 
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""



    #CORS 
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        ]


# Singleton instance used across the app
settings = Settings()
