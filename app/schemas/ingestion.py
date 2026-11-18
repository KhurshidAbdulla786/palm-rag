from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.services.chunking import ChunkingStrategy


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    content_type: str
    size_bytes: int
    num_characters: int
    num_chunks: int
    chunking_strategy: ChunkingStrategy
    chunk_size: int
    chunk_overlap: int
    created_at: datetime
