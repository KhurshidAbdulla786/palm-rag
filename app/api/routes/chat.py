from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import ChatServiceDep, SessionDep
from app.db.repositories import BookingRepository
from app.schemas.chat import BookingResponse, ChatRequest, ChatResponse

router = APIRouter(tags=["Conversational RAG"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, service: ChatServiceDep) -> ChatResponse:
    """Multi-turn conversational RAG with LLM-driven interview booking."""
    return await service.chat(request)


@router.get("/bookings", response_model=list[BookingResponse])
async def list_bookings(
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[BookingResponse]:
    bookings = await BookingRepository(session).list(limit, offset)
    return [BookingResponse.model_validate(b) for b in bookings]
