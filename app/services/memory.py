import json
from typing import Literal, TypedDict

from redis.asyncio import Redis


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class RedisChatMemory:
    """Per-session chat history and in-progress booking state, stored in Redis."""

    def __init__(self, client: Redis, ttl_seconds: int, max_messages: int) -> None:
        self._client = client
        self._ttl = ttl_seconds
        self._max = max_messages

    @staticmethod
    def _history_key(session_id: str) -> str:
        return f"chat:{session_id}:history"

    @staticmethod
    def _booking_key(session_id: str) -> str:
        return f"chat:{session_id}:booking"

    async def get_history(self, session_id: str) -> list[ChatMessage]:
        raw = await self._client.lrange(self._history_key(session_id), 0, -1)
        return [json.loads(item) for item in raw]

    async def append(self, session_id: str, *messages: ChatMessage) -> None:
        key = self._history_key(session_id)
        pipe = self._client.pipeline()
        pipe.rpush(key, *(json.dumps(m) for m in messages))
        pipe.ltrim(key, -self._max, -1)
        pipe.expire(key, self._ttl)
        await pipe.execute()

    async def get_booking_state(self, session_id: str) -> dict[str, str]:
        raw = await self._client.get(self._booking_key(session_id))
        return json.loads(raw) if raw else {}

    async def set_booking_state(self, session_id: str, state: dict[str, str]) -> None:
        await self._client.set(
            self._booking_key(session_id), json.dumps(state), ex=self._ttl
        )

    async def clear_booking_state(self, session_id: str) -> None:
        await self._client.delete(self._booking_key(session_id))
