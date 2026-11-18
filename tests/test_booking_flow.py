import asyncio
from collections.abc import Sequence
from datetime import date, timedelta
from typing import Any

from sqlalchemy import select

from app.db.base import create_engine, create_session_factory, init_models
from app.db.models import Booking
from app.db.repositories import BookingRepository
from app.services.booking import BookingService
from app.services.llm import Message


class FakeLLM:
    async def complete_json(self, messages: Sequence[Message]) -> dict[str, Any]:
        return {
            "wants_booking": True,
            "cancel": False,
            "name": "Asha Rai",
            "email": "asha@example.com",
            "date": (date.today() + timedelta(days=2)).isoformat(),
            "time": "15:30",
        }


class FakeMemory:
    def __init__(self) -> None:
        self.booking_state: dict[str, str] = {}
        self.cleared: list[str] = []

    async def set_booking_state(self, session_id: str, state: dict[str, str]) -> None:
        self.booking_state = state

    async def clear_booking_state(self, session_id: str) -> None:
        self.cleared.append(session_id)
        self.booking_state = {}


def test_llm_extracted_interview_slots_are_persisted_to_sql() -> None:
    async def run() -> None:
        engine = create_engine("sqlite+aiosqlite:///:memory:")
        await init_models(engine)
        session_factory = create_session_factory(engine)
        memory = FakeMemory()
        try:
            async with session_factory() as session:
                service = BookingService(
                    FakeLLM(),  # type: ignore[arg-type]
                    memory,  # type: ignore[arg-type]
                    BookingRepository(session),
                )
                analysis = await service.analyze(
                    (
                        "Book an interview for me; I am Asha Rai, "
                        "asha@example.com, in two days at 3:30pm."
                    ),
                    [],
                    {},
                )
                assert analysis.wants_booking is True
                assert analysis.slots["name"] == "Asha Rai"

                outcome = await service.advance("candidate-1", analysis, {})
                assert outcome.booking is not None
                assert outcome.booking.name == "Asha Rai"
                assert outcome.booking.email == "asha@example.com"
                assert outcome.booking.interview_date == date.today() + timedelta(
                    days=2
                )
                assert (
                    outcome.booking.interview_time.isoformat(timespec="minutes")
                    == "15:30"
                )
                assert memory.cleared == ["candidate-1"]

                saved = (
                    await session.scalars(
                        select(Booking).where(Booking.session_id == "candidate-1")
                    )
                ).one()
                assert saved.id == outcome.booking.id
        finally:
            await engine.dispose()

    asyncio.run(run())
