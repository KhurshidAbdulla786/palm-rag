from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Booking, Document


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, document: Document) -> Document:
        self._session.add(document)
        await self._session.commit()
        await self._session.refresh(document)
        return document

    async def get(self, document_id: str) -> Document | None:
        return await self._session.get(Document, document_id)

    async def list(self, limit: int = 50, offset: int = 0) -> Sequence[Document]:
        stmt = (
            select(Document)
            .order_by(Document.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return (await self._session.scalars(stmt)).all()


class BookingRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, booking: Booking) -> Booking:
        self._session.add(booking)
        await self._session.commit()
        await self._session.refresh(booking)
        return booking

    async def list(self, limit: int = 50, offset: int = 0) -> Sequence[Booking]:
        stmt = (
            select(Booking)
            .order_by(Booking.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return (await self._session.scalars(stmt)).all()
