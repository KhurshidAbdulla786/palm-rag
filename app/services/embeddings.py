from collections.abc import Sequence

from openai import AsyncOpenAI


class EmbeddingService:
    def __init__(self, client: AsyncOpenAI, model: str, batch_size: int = 64) -> None:
        self._client = client
        self._model = model
        self._batch_size = batch_size

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = list(texts[i : i + self._batch_size])
            response = await self._client.embeddings.create(
                model=self._model, input=batch
            )
            vectors.extend(item.embedding for item in response.data)
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]
