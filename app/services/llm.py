import json
from collections.abc import Sequence
from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

Message = ChatCompletionMessageParam


class LLMService:
    """Thin wrapper around the chat-completions API."""

    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    async def complete(
        self, messages: Sequence[Message], temperature: float = 0.2
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=cast(list[ChatCompletionMessageParam], list(messages)),
            temperature=temperature,
        )
        return (response.choices[0].message.content or "").strip()

    async def complete_json(self, messages: Sequence[Message]) -> dict[str, Any]:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=cast(list[ChatCompletionMessageParam], list(messages)),
            temperature=0,
            response_format={"type": "json_object"},
        )
        try:
            parsed = json.loads(response.choices[0].message.content or "{}")
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
