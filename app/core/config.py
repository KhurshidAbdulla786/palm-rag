from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str
    openai_base_url: str | None = None
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "documents"

    redis_url: str = "redis://localhost:6379/0"
    chat_history_ttl_seconds: int = 60 * 60 * 24
    chat_history_max_messages: int = 20

    database_url: str = "sqlite+aiosqlite:///./app.db"

    retrieval_top_k: int = 5
    max_upload_mb: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()
