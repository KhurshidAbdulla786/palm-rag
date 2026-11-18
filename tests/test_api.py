from collections.abc import Iterator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_chat_service, get_container, get_ingestion_service
from app.main import app
from app.schemas.chat import ChatRequest, ChatResponse, Intent
from app.services.chunking import ChunkingStrategy


class FakeIngestionService:
    def __init__(self) -> None:
        self.arguments: dict[str, object] | None = None

    async def ingest(self, **kwargs: object) -> SimpleNamespace:
        self.arguments = kwargs
        data = cast(bytes, kwargs["data"])
        strategy = cast(ChunkingStrategy, kwargs["strategy"])
        return SimpleNamespace(
            id="doc-123",
            filename=kwargs["filename"],
            content_type=kwargs["content_type"],
            size_bytes=len(data),
            num_characters=17,
            num_chunks=1,
            chunking_strategy=strategy.value,
            chunk_size=kwargs["chunk_size"],
            chunk_overlap=kwargs["chunk_overlap"],
            created_at=datetime.now(UTC),
        )


class FakeChatService:
    async def chat(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(
            session_id=request.session_id,
            intent=Intent.RAG,
            answer="Answer from the ingested document.",
        )


@pytest.fixture
def client() -> Iterator[tuple[TestClient, FakeIngestionService]]:
    ingestion = FakeIngestionService()
    overrides = app.dependency_overrides.copy()
    app.dependency_overrides[get_container] = lambda: SimpleNamespace(
        settings=SimpleNamespace(max_upload_mb=1)
    )
    app.dependency_overrides[get_ingestion_service] = lambda: ingestion
    app.dependency_overrides[get_chat_service] = FakeChatService
    test_client = TestClient(app)
    try:
        yield test_client, ingestion
    finally:
        test_client.close()
        app.dependency_overrides.clear()
        app.dependency_overrides.update(overrides)


def test_document_upload_api_returns_metadata(
    client: tuple[TestClient, FakeIngestionService],
) -> None:
    test_client, ingestion = client
    response = test_client.post(
        "/api/v1/documents",
        files={"file": ("guide.txt", b"Some document text.", "text/plain")},
        data={
            "chunking_strategy": "sentence",
            "chunk_size": "800",
            "chunk_overlap": "100",
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["id"] == "doc-123"
    assert payload["filename"] == "guide.txt"
    assert payload["chunking_strategy"] == "sentence"
    assert payload["num_chunks"] == 1
    assert ingestion.arguments is not None
    assert ingestion.arguments["data"] == b"Some document text."


def test_document_upload_api_rejects_file_over_limit(
    client: tuple[TestClient, FakeIngestionService],
) -> None:
    test_client, _ = client
    response = test_client.post(
        "/api/v1/documents",
        files={"file": ("large.txt", b"x" * (1024 * 1024 + 1), "text/plain")},
    )
    assert response.status_code == 413
    assert "exceeds" in response.json()["detail"]


def test_chat_api_returns_conversational_response(
    client: tuple[TestClient, FakeIngestionService],
) -> None:
    test_client, _ = client
    response = test_client.post(
        "/api/v1/chat",
        json={"session_id": "session-1", "message": "Summarize the document"},
    )
    assert response.status_code == 200
    assert response.json()["session_id"] == "session-1"
    assert response.json()["intent"] == "rag"
    assert response.json()["answer"] == "Answer from the ingested document."


def test_chat_api_validates_required_fields(
    client: tuple[TestClient, FakeIngestionService],
) -> None:
    test_client, _ = client
    response = test_client.post("/api/v1/chat", json={"session_id": "session-1"})
    assert response.status_code == 422
