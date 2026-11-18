from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.container import Container
from app.db.repositories import BookingRepository, DocumentRepository
from app.services.booking import BookingService
from app.services.ingestion import IngestionService
from app.services.rag import ConversationalRAGService


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


async def get_db_session(container: ContainerDep) -> AsyncIterator[AsyncSession]:
    async with container.session_factory() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_db_session)]


def get_ingestion_service(
    container: ContainerDep, session: SessionDep
) -> IngestionService:
    return IngestionService(
        embeddings=container.embeddings,
        vector_store=container.vector_store,
        documents=DocumentRepository(session),
    )


def get_chat_service(
    container: ContainerDep, session: SessionDep
) -> ConversationalRAGService:
    booking = BookingService(
        container.llm, container.memory, BookingRepository(session)
    )
    return ConversationalRAGService(
        llm=container.llm,
        embeddings=container.embeddings,
        vector_store=container.vector_store,
        memory=container.memory,
        booking=booking,
        top_k=container.settings.retrieval_top_k,
    )


IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]
ChatServiceDep = Annotated[ConversationalRAGService, Depends(get_chat_service)]
