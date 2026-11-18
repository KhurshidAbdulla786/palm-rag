from collections.abc import Sequence
from typing import cast

from openai.types.chat import ChatCompletionMessageParam

from app.schemas.chat import ChatRequest, ChatResponse, Intent, SourceChunk
from app.services.booking import BookingService
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.services.memory import ChatMessage, RedisChatMemory
from app.services.vector_store import QdrantVectorStore, RetrievedChunk

_CONDENSE_PROMPT = (
    "Rewrite the user's latest message as a standalone question, using the "
    "conversation for context (resolve pronouns and references). If it is already "
    "standalone, return it unchanged. Return ONLY the question."
)
_ANSWER_PROMPT = (
    "You are a helpful assistant answering questions about the user's documents. "
    "Use ONLY the numbered context passages below. If the answer isn't in them, "
    "say you don't know based on the provided documents. Cite passages like [1] "
    "where relevant.\n\nContext:\n{context}"
)
_NO_CONTEXT_REPLY = "I couldn't find anything relevant in the ingested documents."


def _format_context(chunks: Sequence[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{i}] ({c.filename})\n{c.text}" for i, c in enumerate(chunks, start=1)
    )


class ConversationalRAGService:
    """Custom RAG pipeline: condense -> retrieve -> generate, with Redis memory.

    Booking requests are routed to ``BookingService`` instead of retrieval.
    """

    def __init__(
        self,
        llm: LLMService,
        embeddings: EmbeddingService,
        vector_store: QdrantVectorStore,
        memory: RedisChatMemory,
        booking: BookingService,
        top_k: int,
    ) -> None:
        self._llm = llm
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._memory = memory
        self._booking = booking
        self._top_k = top_k

    async def chat(self, request: ChatRequest) -> ChatResponse:
        session_id = request.session_id
        history = await self._memory.get_history(session_id)
        state = await self._memory.get_booking_state(session_id)

        analysis = await self._booking.analyze(request.message, history, state)
        in_booking_flow = analysis.wants_booking or (bool(state) and analysis.cancel)

        if in_booking_flow:
            outcome = await self._booking.advance(session_id, analysis, state)
            response = ChatResponse(
                session_id=session_id,
                intent=Intent.BOOKING,
                answer=outcome.reply,
                booking=outcome.booking,
            )
        else:
            answer, sources = await self._answer_question(request, history)
            response = ChatResponse(
                session_id=session_id, intent=Intent.RAG, answer=answer, sources=sources
            )

        await self._memory.append(
            session_id,
            ChatMessage(role="user", content=request.message),
            ChatMessage(role="assistant", content=response.answer),
        )
        return response

    async def _answer_question(
        self, request: ChatRequest, history: Sequence[ChatMessage]
    ) -> tuple[str, list[SourceChunk]]:
        query = await self._condense(request.message, history)
        vector = await self._embeddings.embed_query(query)
        chunks = await self._vector_store.search(
            vector, self._top_k, request.document_id
        )
        if not chunks:
            return _NO_CONTEXT_REPLY, []

        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": _ANSWER_PROMPT.format(context=_format_context(chunks)),
            },
            *[
                cast(
                    ChatCompletionMessageParam,
                    {"role": message["role"], "content": message["content"]},
                )
                for message in history
            ],
            {"role": "user", "content": request.message},
        ]
        answer = await self._llm.complete(messages)
        sources = [
            SourceChunk(
                document_id=c.document_id,
                filename=c.filename,
                chunk_index=c.chunk_index,
                score=c.score,
                text=c.text,
            )
            for c in chunks
        ]
        return answer, sources

    async def _condense(self, message: str, history: Sequence[ChatMessage]) -> str:
        if not history:
            return message
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": _CONDENSE_PROMPT},
            {
                "role": "user",
                "content": f"Conversation:\n{transcript}\n\nLatest message: {message}",
            },
        ]
        return await self._llm.complete(messages, temperature=0)
