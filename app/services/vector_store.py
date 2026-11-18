import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

_UPSERT_BATCH = 100


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    document_id: str
    filename: str
    chunk_index: int
    text: str
    score: float


def _document_filter(document_id: str) -> Filter:
    return Filter(
        must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
    )


class QdrantVectorStore:
    def __init__(self, client: AsyncQdrantClient, collection: str, dim: int) -> None:
        self._client = client
        self._collection = collection
        self._dim = dim

    async def ensure_collection(self) -> None:
        if not await self._client.collection_exists(self._collection):
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            )
        await self._client.create_payload_index(
            collection_name=self._collection,
            field_name="document_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    async def upsert_chunks(
        self,
        *,
        document_id: str,
        filename: str,
        chunks: Sequence[str],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=list(vector),
                payload={
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": index,
                    "text": text,
                },
            )
            for index, (text, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        for i in range(0, len(points), _UPSERT_BATCH):
            await self._client.upsert(
                collection_name=self._collection, points=points[i : i + _UPSERT_BATCH]
            )

    async def search(
        self, vector: Sequence[float], top_k: int, document_id: str | None = None
    ) -> list[RetrievedChunk]:
        response = await self._client.query_points(
            collection_name=self._collection,
            query=list(vector),
            limit=top_k,
            query_filter=_document_filter(document_id) if document_id else None,
            with_payload=True,
        )
        results: list[RetrievedChunk] = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                RetrievedChunk(
                    document_id=str(payload.get("document_id", "")),
                    filename=str(payload.get("filename", "")),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    text=str(payload.get("text", "")),
                    score=float(point.score),
                )
            )
        return results

    async def delete_document(self, document_id: str) -> None:
        await self._client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(filter=_document_filter(document_id)),
        )
