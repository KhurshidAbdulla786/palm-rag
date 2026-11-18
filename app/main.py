import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from openai import AsyncOpenAI, OpenAIError
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from app.api.routes import chat, ingestion
from app.core.config import get_settings
from app.core.container import Container
from app.core.exceptions import AppError
from app.db.base import create_engine, create_session_factory, init_models
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.services.memory import RedisChatMemory
from app.services.vector_store import QdrantVectorStore

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    engine = create_engine(settings.database_url)
    await init_models(engine)

    openai_client = AsyncOpenAI(
        api_key=settings.openai_api_key, base_url=settings.openai_base_url
    )
    redis_client: Redis = Redis.from_url(settings.redis_url, decode_responses=True)
    qdrant_client = AsyncQdrantClient(
        url=settings.qdrant_url, api_key=settings.qdrant_api_key
    )

    vector_store = QdrantVectorStore(
        qdrant_client, settings.qdrant_collection, settings.embedding_dim
    )
    await vector_store.ensure_collection()

    app.state.container = Container(
        settings=settings,
        llm=LLMService(openai_client, settings.llm_model),
        embeddings=EmbeddingService(openai_client, settings.embedding_model),
        vector_store=vector_store,
        memory=RedisChatMemory(
            redis_client,
            settings.chat_history_ttl_seconds,
            settings.chat_history_max_messages,
        ),
        session_factory=create_session_factory(engine),
    )
    try:
        yield
    finally:
        await redis_client.aclose()
        await qdrant_client.close()
        await engine.dispose()


app = FastAPI(title="Palm Mind RAG Backend", version="1.0.0", lifespan=lifespan)
app.include_router(ingestion.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")


@app.exception_handler(AppError)
async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.exception_handler(OpenAIError)
async def handle_openai_error(_: Request, exc: OpenAIError) -> JSONResponse:
    return JSONResponse(
        status_code=502, content={"detail": f"LLM provider error: {exc}"}
    )


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Send the browser root URL to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
