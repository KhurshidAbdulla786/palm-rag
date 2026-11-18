import asyncio
import logging
import uuid

from app.core.exceptions import EmptyDocumentError
from app.db.models import Document
from app.db.repositories import DocumentRepository
from app.services.chunking import ChunkingStrategy, build_chunker
from app.services.embeddings import EmbeddingService
from app.services.text_extraction import extract_text
from app.services.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self,
        embeddings: EmbeddingService,
        vector_store: QdrantVectorStore,
        documents: DocumentRepository,
    ) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._documents = documents

    async def ingest(
        self,
        *,
        filename: str,
        content_type: str,
        data: bytes,
        strategy: ChunkingStrategy,
        chunk_size: int,
        chunk_overlap: int,
    ) -> Document:
        chunker = build_chunker(
            strategy, chunk_size, chunk_overlap
        )  # validates config first
        text = await asyncio.to_thread(extract_text, filename, data)
        if not text.strip():
            raise EmptyDocumentError("No extractable text found in the document")

        chunks = chunker.split(text)
        if not chunks:
            raise EmptyDocumentError("Document produced no chunks")

        vectors = await self._embeddings.embed_documents(chunks)
        document_id = str(uuid.uuid4())
        try:
            await self._vector_store.upsert_chunks(
                document_id=document_id,
                filename=filename,
                chunks=chunks,
                vectors=vectors,
            )
            return await self._documents.add(
                Document(
                    id=document_id,
                    filename=filename,
                    content_type=content_type,
                    size_bytes=len(data),
                    num_characters=len(text),
                    num_chunks=len(chunks),
                    chunking_strategy=strategy.value,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            )
        except Exception:
            logger.exception(
                "Ingestion failed; rolling back vectors for %s", document_id
            )
            try:
                await self._vector_store.delete_document(document_id)
            except Exception:
                logger.exception("Vector rollback failed for document %s", document_id)
            raise
