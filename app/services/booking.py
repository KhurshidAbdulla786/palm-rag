import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from pydantic import EmailStr, TypeAdapter, ValidationError

from app.db.models import Booking
from app.db.repositories import BookingRepository
from app.services.llm import LLMService
from app.services.memory import ChatMessage, RedisChatMemory

REQUIRED_FIELDS: tuple[str, ...] = ("name", "email", "date", "time")
_FIELD_LABELS: dict[str, str] = {
    "name": "your full name",
    "email": "your email address",
    "date": "your preferred interview date",
    "time": "your preferred interview time",
}
_email_adapter: TypeAdapter[str] = TypeAdapter(EmailStr)

_ANALYZER_PROMPT = """You analyse the latest message in a chat with a document Q&A \
assistant that can also book interviews. Today's date is {today}.

Return ONLY a JSON object with these keys:
- "wants_booking": true if the user wants to schedule/book an interview, or is \
supplying details for a booking already in progress; false for ordinary questions.
- "cancel": true only if the user abandons a booking in progress.
- "name": the user's full name if stated, else null.
- "email": the user's email if stated, else null.
- "date": interview date as YYYY-MM-DD (resolve relative dates like "tomorrow" or \
"next Monday" using today's date), else null.
- "time": interview time in 24-hour HH:MM, else null.
Only extract values the user actually stated. Never invent or guess missing values."""


@dataclass(frozen=True, slots=True)
class TurnAnalysis:
    wants_booking: bool
    cancel: bool
    slots: dict[str, str | None]


@dataclass(frozen=True, slots=True)
class BookingOutcome:
    reply: str
    booking: Booking | None = None


def _normalize_slot(field: str, value: str, today: date) -> str:
    if field == "name":
        if len(value) < 2:
            raise ValueError("The name looks too short")
        return value
    if field == "email":
        try:
            return _email_adapter.validate_python(value)
        except ValidationError as exc:
            raise ValueError(f"'{value}' isn't a valid email address") from exc
    if field == "date":
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"I couldn't understand the date '{value}'") from exc
        if parsed < today:
            raise ValueError("The interview date can't be in the past")
        return parsed.isoformat()
    if field == "time":
        try:
            return datetime.strptime(value, "%H:%M").strftime("%H:%M")
        except ValueError as exc:
            raise ValueError(f"I couldn't understand the time '{value}'") from exc
    raise ValueError(f"Unknown field {field}")


def merge_and_validate(
    current: Mapping[str, str], extracted: Mapping[str, Any], today: date
) -> tuple[dict[str, str], list[str]]:
    """Merge newly extracted slot values into the stored state, validating each."""
    merged = dict(current)
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        raw = extracted.get(field)
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            merged[field] = _normalize_slot(field, raw.strip(), today)
        except ValueError as exc:
            errors.append(str(exc))
    return merged, errors


def _join(items: Sequence[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


class BookingService:
    """LLM-driven slot filling for interview bookings, persisted via SQL."""

    def __init__(
        self, llm: LLMService, memory: RedisChatMemory, repository: BookingRepository
    ) -> None:
        self._llm = llm
        self._memory = memory
        self._repository = repository

    async def analyze(
        self, message: str, history: Sequence[ChatMessage], state: Mapping[str, str]
    ) -> TurnAnalysis:
        payload = {
            "booking_in_progress": bool(state),
            "collected_so_far": dict(state),
            "recent_history": list(history[-6:]),
            "latest_message": message,
        }
        result = await self._llm.complete_json(
            [
                {
                    "role": "system",
                    "content": _ANALYZER_PROMPT.format(today=date.today().isoformat()),
                },
                {"role": "user", "content": json.dumps(payload)},
            ]
        )
        return TurnAnalysis(
            wants_booking=bool(result.get("wants_booking")),
            cancel=bool(result.get("cancel")),
            slots={f: result.get(f) for f in REQUIRED_FIELDS},
        )

    async def advance(
        self, session_id: str, analysis: TurnAnalysis, state: Mapping[str, str]
    ) -> BookingOutcome:
        if analysis.cancel:
            await self._memory.clear_booking_state(session_id)
            return BookingOutcome("No problem, I've cancelled the booking request.")

        merged, errors = merge_and_validate(state, analysis.slots, date.today())
        missing = [f for f in REQUIRED_FIELDS if f not in merged]

        if missing:
            await self._memory.set_booking_state(session_id, merged)
            parts: list[str] = []
            if errors:
                parts.append("; ".join(errors) + ".")
            parts.append(
                "To book your interview I still need "
                + _join([_FIELD_LABELS[f] for f in missing])
                + "."
            )
            return BookingOutcome(" ".join(parts))

        booking = await self._repository.add(
            Booking(
                session_id=session_id,
                name=merged["name"],
                email=merged["email"],
                interview_date=date.fromisoformat(merged["date"]),
                interview_time=datetime.strptime(merged["time"], "%H:%M").time(),
            )
        )
        await self._memory.clear_booking_state(session_id)
        return BookingOutcome(
            f"Your interview is booked for {merged['date']} at {merged['time']}. "
            f"Booking email: {booking.email}.",
            booking,
        )
