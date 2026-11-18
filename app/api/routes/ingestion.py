from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status

from app.api.deps import ContainerDep, IngestionServiceDep, SessionDep
from app.core.exceptions import FileTooLargeError
from app.db.repositories import DocumentRepository
from app.schemas.ingestion import DocumentResponse
from app.services.chunking import ChunkingStrategy

router = APIRouter(prefix="/documents", tags=["Document Ingestion"])


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(
    service: IngestionServiceDep,
    container: ContainerDep,
    file: Annotated[UploadFile, File(description=".pdf or .txt file")],
    chunking_strategy: Annotated[ChunkingStrategy, Form()] = ChunkingStrategy.FIXED,
    chunk_size: Annotated[int, Form(ge=100, le=8000)] = 1000,
    chunk_overlap: Annotated[int, Form(ge=0, le=2000)] = 200,
) -> DocumentResponse:
    """Upload a document, chunk it, embed it, and store vectors + metadata."""
    limit = container.settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise FileTooLargeError(
            f"File exceeds {container.settings.max_upload_mb} MB limit"
        )

    document = await service.ingest(
        filename=file.filename or "upload",
        content_type=file.content_type or "application/octet-stream",
        data=data,
        strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return DocumentResponse.model_validate(document)


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[DocumentResponse]:
    docs = await DocumentRepository(session).list(limit, offset)
    return [DocumentResponse.model_validate(d) for d in docs]


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str, session: SessionDep) -> DocumentResponse:
    document = await DocumentRepository(session).get(document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return DocumentResponse.model_validate(document)
