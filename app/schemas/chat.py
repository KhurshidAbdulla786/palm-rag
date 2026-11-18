from datetime import date, datetime, time
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Intent(StrEnum):
    RAG = "rag"
    BOOKING = "booking"


class ChatRequest(BaseModel):
    session_id: str = Field(
        min_length=1, max_length=128, description="Client-chosen conversation id"
    )
    message: str = Field(min_length=1, max_length=4000)
    document_id: str | None = Field(
        default=None,
        description="Optionally restrict retrieval to a single ingested document",
    )


class SourceChunk(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    score: float
    text: str


class BookingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    name: str
    email: str
    interview_date: date
    interview_time: time
    created_at: datetime


class ChatResponse(BaseModel):
    session_id: str
    intent: Intent
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)
    booking: BookingResponse | None = None
