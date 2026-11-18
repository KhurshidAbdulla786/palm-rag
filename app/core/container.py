from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.services.memory import RedisChatMemory
from app.services.vector_store import QdrantVectorStore


@dataclass(frozen=True, slots=True)
class Container:
    """Long-lived singletons created at startup and shared across requests."""

    settings: Settings
    llm: LLMService
    embeddings: EmbeddingService
    vector_store: QdrantVectorStore
    memory: RedisChatMemory
    session_factory: async_sessionmaker[AsyncSession]
