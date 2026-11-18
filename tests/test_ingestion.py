import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from app.core.exceptions import EmptyDocumentError, UnsupportedFileTypeError
from app.db.models import Document
from app.services.chunking import ChunkingStrategy
from app.services.ingestion import IngestionService


class FakeEmbeddings:
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(index)] for index, _ in enumerate(texts)]


class FakeVectorStore:
    def __init__(self) -> None:
        self.upserted: dict[str, object] | None = None
        self.deleted: list[str] = []

    async def upsert_chunks(
        self,
        *,
        document_id: str,
        filename: str,
        chunks: list[str],
        vectors: list[list[float]],
    ) -> None:
        self.upserted = {
            "document_id": document_id,
            "filename": filename,
            "chunks": chunks,
            "vectors": vectors,
        }

    async def delete_document(self, document_id: str) -> None:
        self.deleted.append(document_id)


class FakeDocumentRepository:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.saved: Document | None = None

    async def add(self, document: Document) -> Document:
        if self.fail:
            raise RuntimeError("database unavailable")
        document.created_at = datetime.now(UTC)
        self.saved = document
        return document


def make_service(
    *, fail_metadata: bool = False
) -> tuple[IngestionService, FakeVectorStore, FakeDocumentRepository]:
    vectors = FakeVectorStore()
    documents = FakeDocumentRepository(fail=fail_metadata)
    service = IngestionService(FakeEmbeddings(), vectors, documents)  # type: ignore[arg-type]
    return service, vectors, documents


def test_ingests_txt_and_persists_document_metadata() -> None:
    async def run() -> None:
        service, vectors, documents = make_service()
        document = await service.ingest(
            filename="guide.txt",
            content_type="text/plain",
            data=b"A first sentence. A second sentence.",
            strategy=ChunkingStrategy.SENTENCE,
            chunk_size=100,
            chunk_overlap=10,
        )

        assert document.filename == "guide.txt"
        assert document.size_bytes == len(b"A first sentence. A second sentence.")
        assert document.num_chunks == 1
        assert document.chunking_strategy == "sentence"
        assert vectors.upserted is not None
        assert vectors.upserted["document_id"] == document.id
        assert vectors.upserted["chunks"] == ["A first sentence. A second sentence."]
        assert documents.saved is document
        assert vectors.deleted == []

    asyncio.run(run())


def test_ingestion_rejects_empty_documents() -> None:
    async def run() -> None:
        service, vectors, documents = make_service()
        with pytest.raises(EmptyDocumentError):
            await service.ingest(
                filename="empty.txt",
                content_type="text/plain",
                data=b"  \n  ",
                strategy=ChunkingStrategy.FIXED,
                chunk_size=100,
                chunk_overlap=10,
            )
        assert vectors.upserted is None
        assert documents.saved is None

    asyncio.run(run())


def test_ingestion_rejects_unsupported_extension_before_embedding() -> None:
    async def run() -> None:
        service, vectors, documents = make_service()
        with pytest.raises(UnsupportedFileTypeError):
            await service.ingest(
                filename="notes.docx",
                content_type="application/octet-stream",
                data=b"not supported",
                strategy=ChunkingStrategy.FIXED,
                chunk_size=100,
                chunk_overlap=10,
            )
        assert vectors.upserted is None
        assert documents.saved is None

    asyncio.run(run())


def test_ingestion_rolls_back_vectors_if_metadata_save_fails() -> None:
    async def run() -> None:
        service, vectors, _ = make_service(fail_metadata=True)
        with pytest.raises(RuntimeError, match="database unavailable"):
            await service.ingest(
                filename="guide.txt",
                content_type="text/plain",
                data=b"Text that should be indexed.",
                strategy=ChunkingStrategy.FIXED,
                chunk_size=100,
                chunk_overlap=10,
            )
        assert vectors.upserted is not None
        assert vectors.deleted == [str(vectors.upserted["document_id"])]

    asyncio.run(run())
