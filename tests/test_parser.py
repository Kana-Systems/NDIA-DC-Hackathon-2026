import io
import os
import time
import zipfile

import pytest
from docx import Document

from app.config import Settings
from app.parsers import DocumentParser


def _hanging_parse_worker(
    _filename: str,
    _data: bytes,
    _settings: Settings,
    _connection: object,
) -> None:
    time.sleep(30)


def _crashing_parse_worker(
    _filename: str,
    _data: bytes,
    _settings: Settings,
    _connection: object,
) -> None:
    os._exit(9)


def _docx_bytes(*paragraphs: str) -> bytes:
    document = Document()
    for text in paragraphs:
        document.add_paragraph(text)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_docx_parser_preserves_paragraph_locations() -> None:
    parsed = DocumentParser(Settings()).parse(
        "contract.docx", _docx_bytes("First clause.", "Second clause.")
    )

    assert [segment.text for segment in parsed.segments] == [
        "First clause.",
        "Second clause.",
    ]
    assert parsed.segments[1].location.paragraph == 2
    assert parsed.segments[1].location.label == "Page 1, paragraph 2"
    assert len(parsed.sha256) == 64


def test_pdf_parser_preserves_page_and_paragraph_locations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Page:
        def __init__(self, text: str) -> None:
            self.text = text

        def extract_text(self) -> str:
            return self.text

    class Reader:
        def __init__(self, _stream: object) -> None:
            self.pages = [Page("First paragraph.\n\nSecond paragraph."), Page("Third.")]

    monkeypatch.setattr("app.parsers.PdfReader", Reader)
    parsed = DocumentParser(Settings()).parse("contract.pdf", b"%PDF-synthetic")

    assert [segment.location.page for segment in parsed.segments] == [1, 1, 2]
    assert parsed.segments[1].segment_id == "p1-para2"


def test_parser_rejects_unsupported_and_empty_documents() -> None:
    parser = DocumentParser(Settings())
    with pytest.raises(ValueError, match="Unsupported"):
        parser.parse("contract.txt", b"text")
    with pytest.raises(ValueError, match="no extractable"):
        parser.parse("empty.docx", _docx_bytes())


def test_docx_preflight_rejects_excess_members_and_archive_expansion() -> None:
    many_members = io.BytesIO()
    with zipfile.ZipFile(many_members, "w") as archive:
        for index in range(11):
            archive.writestr(f"item-{index}.xml", "x")
    with pytest.raises(ValueError, match="member limit"):
        DocumentParser(Settings(max_docx_members=10)).parse(
            "many.docx",
            many_members.getvalue(),
        )

    expanded = io.BytesIO()
    with zipfile.ZipFile(expanded, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("word/document.xml", b"x" * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match="uncompressed size"):
        DocumentParser(
            Settings(
                max_docx_uncompressed_mb=1,
                max_docx_compression_ratio=1_000,
            )
        ).parse("expanded.docx", expanded.getvalue())


def test_pdf_page_limit_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    class Reader:
        def __init__(self, _stream: object) -> None:
            self.pages = [object(), object()]

    monkeypatch.setattr("app.parsers.PdfReader", Reader)
    with pytest.raises(ValueError, match="page limit"):
        DocumentParser(Settings(max_pdf_pages=1)).parse("large.pdf", b"%PDF")


def test_text_and_segment_limits_are_enforced() -> None:
    with pytest.raises(ValueError, match="text limit"):
        DocumentParser(
            Settings(
                max_extracted_characters=1_000,
                max_docx_compression_ratio=1_000,
            )
        ).parse("text.docx", _docx_bytes("x" * 1_001))

    with pytest.raises(ValueError, match="segment limit"):
        DocumentParser(Settings(max_document_segments=10)).parse(
            "segments.docx",
            _docx_bytes(*[f"Paragraph {index}." for index in range(11)]),
        )


def test_isolated_parser_terminates_on_timeout() -> None:
    parser = DocumentParser(
        Settings(parse_timeout_seconds=0.1),
        worker_target=_hanging_parse_worker,
    )

    started = time.monotonic()
    with pytest.raises(ValueError, match="timed out"):
        parser.parse_isolated("contract.docx", b"ignored")

    assert time.monotonic() - started < 5


def test_isolated_parser_sanitizes_child_crash() -> None:
    parser = DocumentParser(
        Settings(parse_timeout_seconds=5),
        worker_target=_crashing_parse_worker,
    )

    with pytest.raises(ValueError, match="failed in isolated worker") as error:
        parser.parse_isolated("secret-name.docx", b"secret-content")

    assert "secret" not in str(error.value)


def test_isolation_runner_is_injectable_without_process() -> None:
    settings = Settings()
    direct_parser = DocumentParser(settings)
    parser = DocumentParser(settings, isolation_runner=direct_parser.parse)

    parsed = parser.parse_isolated("contract.docx", _docx_bytes("Clause text."))

    assert parsed.segments[0].text == "Clause text."
