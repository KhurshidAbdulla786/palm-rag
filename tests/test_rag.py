import asyncio
from collections.abc import Sequence

from app.schemas.chat import ChatRequest, Intent
from app.services.booking import BookingOutcome, TurnAnalysis
from app.services.llm import Message
from app.services.memory import ChatMessage
from app.services.rag import ConversationalRAGService
from app.services.vector_store import RetrievedChunk


class FakeMemory:
    def __init__(self) -> None:
        self.history: list[ChatMessage] = []
        self.appended: list[ChatMessage] = []

    async def get_history(self, session_id: str) -> list[ChatMessage]:
        return list(self.history)

    async def get_booking_state(self, session_id: str) -> dict[str, str]:
        return {}

    async def append(self, session_id: str, *messages: ChatMessage) -> None:
        self.appended.extend(messages)
        self.history.extend(messages)


class FakeLLM:
    def __init__(self, answers: list[str]) -> None:
        self.answers = iter(answers)
        self.calls: list[Sequence[Message]] = []

    async def complete(
        self, messages: Sequence[Message], temperature: float = 0.2
    ) -> str:
        self.calls.append(messages)
        return next(self.answers)


class FakeEmbeddings:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.1, 0.2]


class FakeVectorStore:
    def __init__(self) -> None:
        self.searches: list[tuple[list[float], int, str | None]] = []

    async def search(
        self, vector: list[float], top_k: int, document_id: str | None = None
    ) -> list[RetrievedChunk]:
        self.searches.append((vector, top_k, document_id))
        return [
            RetrievedChunk(
                document_id="doc-1",
                filename="guide.txt",
                chunk_index=0,
                text="Refunds are available within 30 days.",
                score=0.92,
            )
        ]


class FakeBooking:
    async def analyze(
        self, message: str, history: Sequence[ChatMessage], state: dict[str, str]
    ) -> TurnAnalysis:
        return TurnAnalysis(wants_booking=False, cancel=False, slots={})

    async def advance(
        self, session_id: str, analysis: TurnAnalysis, state: dict[str, str]
    ) -> BookingOutcome:
        raise AssertionError("RAG questions should not advance a booking")


def test_rag_handles_multi_turn_follow_up_and_keeps_history() -> None:
    async def run() -> None:
        memory = FakeMemory()
        llm = FakeLLM(
            [
                "Refunds are available within 30 days.",
                "What about international orders?",
                "International orders have the same refund window.",
            ]
        )
        embeddings = FakeEmbeddings()
        vector_store = FakeVectorStore()
        service = ConversationalRAGService(
            llm=llm,  # type: ignore[arg-type]
            embeddings=embeddings,  # type: ignore[arg-type]
            vector_store=vector_store,  # type: ignore[arg-type]
            memory=memory,  # type: ignore[arg-type]
            booking=FakeBooking(),  # type: ignore[arg-type]
            top_k=3,
        )

        first = await service.chat(
            ChatRequest(
                session_id="session-1",
                message="What is the refund window?",
                document_id="doc-1",
            )
        )
        second = await service.chat(
            ChatRequest(
                session_id="session-1",
                message="And for international orders?",
                document_id="doc-1",
            )
        )

        assert first.intent is Intent.RAG
        assert first.answer == "Refunds are available within 30 days."
        assert first.sources[0].document_id == "doc-1"
        assert second.answer == "International orders have the same refund window."
        assert len(memory.history) == 4
        assert memory.history[0]["content"] == "What is the refund window?"
        assert embeddings.queries == [
            "What is the refund window?",
            "What about international orders?",
        ]
        assert vector_store.searches == [([0.1, 0.2], 3, "doc-1")] * 2
        assert (
            len(llm.calls) == 3
        )  # first answer; then condense and answer the follow-up

    asyncio.run(run())
