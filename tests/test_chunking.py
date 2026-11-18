import pytest

from app.core.exceptions import InvalidChunkingConfigError
from app.services.chunking import (
    ChunkingStrategy,
    FixedSizeChunker,
    SentenceChunker,
    build_chunker,
)


def test_fixed_chunks_respect_size_and_overlap() -> None:
    text = "abcdefghij" * 30  # 300 chars
    chunks = FixedSizeChunker(100, 20).split(text)
    assert all(len(c) <= 100 for c in chunks)
    assert chunks[0][-20:] == chunks[1][:20]


def test_sentence_chunks_keep_sentences_whole() -> None:
    text = " ".join(f"Sentence number {i} is here." for i in range(40))
    chunks = SentenceChunker(120, 30).split(text)
    assert len(chunks) > 1
    assert all(len(c) <= 120 for c in chunks)
    assert all(c.endswith(".") for c in chunks)


def test_sentence_chunker_handles_giant_sentence() -> None:
    chunks = SentenceChunker(50, 10).split("x" * 200)
    assert chunks and all(len(c) <= 50 for c in chunks)


def test_invalid_overlap_rejected() -> None:
    with pytest.raises(InvalidChunkingConfigError):
        build_chunker(ChunkingStrategy.FIXED, 100, 100)
