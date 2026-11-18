import re
from enum import StrEnum
from typing import Protocol

from app.core.exceptions import InvalidChunkingConfigError

_WHITESPACE_RE = re.compile(r"[ \t]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


class ChunkingStrategy(StrEnum):
    FIXED = "fixed"
    SENTENCE = "sentence"


class Chunker(Protocol):
    def split(self, text: str) -> list[str]: ...


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _validate(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise InvalidChunkingConfigError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise InvalidChunkingConfigError("chunk_overlap must be >= 0 and < chunk_size")


class FixedSizeChunker:
    """Sliding window over characters: simple, predictable chunk sizes."""

    def __init__(self, chunk_size: int, overlap: int) -> None:
        _validate(chunk_size, overlap)
        self._size = chunk_size
        self._overlap = overlap

    def split(self, text: str) -> list[str]:
        text = _normalize(text)
        step = self._size - self._overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            piece = text[start : start + self._size].strip()
            if piece:
                chunks.append(piece)
            if start + self._size >= len(text):
                break
        return chunks


class SentenceChunker:
    """Packs whole sentences/paragraphs up to ``chunk_size`` chars.

    Keeps semantic units intact and carries the last few sentences into the next
    chunk (up to ``overlap`` chars) so context isn't lost at boundaries.
    """

    def __init__(self, chunk_size: int, overlap: int) -> None:
        _validate(chunk_size, overlap)
        self._size = chunk_size
        self._overlap = overlap

    def split(self, text: str) -> list[str]:
        sentences = [_normalize(s) for s in _SENTENCE_RE.split(text) if s and s.strip()]
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for sentence in sentences:
            if (
                len(sentence) > self._size
            ):  # pathological sentence: fall back to hard split
                if current:
                    chunks.append(" ".join(current))
                    current, current_len = [], 0
                chunks.extend(
                    FixedSizeChunker(self._size, self._overlap).split(sentence)
                )
                continue

            if current and current_len + 1 + len(sentence) > self._size:
                chunks.append(" ".join(current))
                current = self._overlap_tail(current)
                while (
                    current and len(" ".join(current)) + 1 + len(sentence) > self._size
                ):
                    current.pop(0)
                current_len = len(" ".join(current))

            current_len += len(sentence) + (1 if current else 0)
            current.append(sentence)

        if current:
            chunks.append(" ".join(current))
        return chunks

    def _overlap_tail(self, sentences: list[str]) -> list[str]:
        tail: list[str] = []
        total = 0
        for sentence in reversed(sentences):
            total += len(sentence) + (1 if tail else 0)
            if total > self._overlap:
                break
            tail.insert(0, sentence)
        return tail


def build_chunker(strategy: ChunkingStrategy, chunk_size: int, overlap: int) -> Chunker:
    if strategy is ChunkingStrategy.FIXED:
        return FixedSizeChunker(chunk_size, overlap)
    if strategy is ChunkingStrategy.SENTENCE:
        return SentenceChunker(chunk_size, overlap)
    raise InvalidChunkingConfigError(f"Unknown chunking strategy: {strategy}")
