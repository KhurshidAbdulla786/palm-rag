from app.core.exceptions import TextExtractionError
from app.services.text_extraction import extract_text


def _minimal_text_pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(pdf)


def test_extracts_text_from_pdf() -> None:
    assert extract_text("guide.pdf", _minimal_text_pdf("Hello PDF")) == "Hello PDF"


def test_invalid_pdf_raises_actionable_extraction_error() -> None:
    try:
        extract_text("broken.pdf", b"not a pdf")
    except TextExtractionError as exc:
        assert "Could not read PDF" in str(exc)
    else:
        raise AssertionError("Invalid PDF should fail text extraction")


def test_txt_extraction_handles_utf8_bom_and_legacy_text() -> None:
    assert extract_text("utf8.txt", b"\xef\xbb\xbfHello") == "Hello"
    assert extract_text("legacy.txt", b"caf\xe9") == "caf\xe9"
