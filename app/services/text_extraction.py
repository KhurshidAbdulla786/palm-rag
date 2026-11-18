import io
from pathlib import PurePath

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from app.core.exceptions import TextExtractionError, UnsupportedFileTypeError

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt"})


def extract_text(filename: str, data: bytes) -> str:
    """Extract plain text from a .pdf or .txt payload."""
    suffix = PurePath(filename).suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(data)
    if suffix == ".txt":
        return _extract_txt(data)
    raise UnsupportedFileTypeError(
        f"Unsupported file type '{suffix or filename}'. Allowed: .pdf, .txt"
    )


def _extract_pdf(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except (PyPdfError, ValueError, OSError) as exc:
        raise TextExtractionError(f"Could not read PDF: {exc}") from exc
    return "\n\n".join(p.strip() for p in pages if p.strip())


def _extract_txt(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")
