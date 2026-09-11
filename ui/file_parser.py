
from __future__ import annotations

import csv
import io
from dataclasses import dataclass

# Word count per synthetic page for formats with no inherent pagination.
# Independent of the chunker's window -- this only decides citation
# granularity and the unit of incremental re-indexing.
WORDS_PER_SYNTHETIC_PAGE = 400


class UnsupportedFileType(Exception):
    pass


class EmptyDocument(Exception):
    """Parsed successfully but yielded no text worth indexing."""


@dataclass
class ParsedFile:
    pages: list[dict]
    page_unit: str  # what one page means for this format, for the UI to show
    note: str | None = None


SUPPORTED_EXTENSIONS = [
    "pdf", "docx", "pptx", "xlsx", "txt", "md", "csv", "html", "htm", "json", "log",
]


def parse_file(filename: str, data: bytes) -> ParsedFile:
    """Dispatch on extension. `data` must be the full file bytes."""
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if extension == "pdf":
        return _parse_pdf(data)
    if extension == "docx":
        return _parse_docx(data)
    if extension == "pptx":
        return _parse_pptx(data)
    if extension == "xlsx":
        return _parse_xlsx(data)
    if extension in ("html", "htm"):
        return _parse_html(data)
    if extension == "csv":
        return _parse_csv(data)
    if extension in ("txt", "md", "json", "log"):
        return _parse_text(data)

    raise UnsupportedFileType(
        f"'.{extension}' is not supported. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
    )


# --- helpers -----------------------------------------------------------------


def _decode(data: bytes) -> str:
    """Best-effort decode. Replaces undecodable bytes rather than failing."""
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return data.decode("utf-8", errors="replace")


def _paginate(text: str, words_per_page: int = WORDS_PER_SYNTHETIC_PAGE) -> list[str]:
    """Split into fixed word blocks. Deterministic for the same input."""
    words = text.split()
    if not words:
        return []
    return [
        " ".join(words[i : i + words_per_page])
        for i in range(0, len(words), words_per_page)
    ]


def _as_pages(blocks: list[str]) -> list[dict]:
    """Number the non-empty blocks from 1, keeping their original order."""
    pages = []
    for block in blocks:
        if block and block.strip():
            pages.append({"page_number": len(pages) + 1, "content": block.strip()})
    return pages


# --- format handlers ---------------------------------------------------------


def _parse_pdf(data: bytes) -> ParsedFile:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    raw = [(page.extract_text() or "") for page in reader.pages]
    pages = _as_pages(raw)

    if not pages:
        raise EmptyDocument(
            f"No selectable text found across {len(raw)} page(s). This is most "
            "likely a scanned PDF -- the pages are images, so there is nothing "
            "to extract without OCR."
        )

    note = None
    if len(pages) < len(raw):
        note = f"{len(raw) - len(pages)} of {len(raw)} page(s) had no text and were skipped."

    return ParsedFile(pages=pages, page_unit="PDF page", note=note)


def _parse_docx(data: bytes) -> ParsedFile:
    import docx

    document = docx.Document(io.BytesIO(data))
    parts = [p.text for p in document.paragraphs if p.text.strip()]

    # Tables carry real content in most business documents; losing them
    # silently would make the answers wrong rather than merely incomplete.
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    text = "\n".join(parts)
    pages = _as_pages(_paginate(text))
    if not pages:
        raise EmptyDocument("The document contains no readable text.")

    return ParsedFile(
        pages=pages,
        page_unit=f"~{WORDS_PER_SYNTHETIC_PAGE}-word block",
        note="Word files carry no page markers, so text was split into even blocks.",
    )


def _parse_pptx(data: bytes) -> ParsedFile:
    from pptx import Presentation

    presentation = Presentation(io.BytesIO(data))
    raw = []
    for slide in presentation.slides:
        parts = []
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                parts.append(shape.text_frame.text.strip())
        raw.append("\n".join(parts))

    pages = _as_pages(raw)
    if not pages:
        raise EmptyDocument("No text found on any slide.")

    return ParsedFile(pages=pages, page_unit="slide")


def _parse_xlsx(data: bytes) -> ParsedFile:
    from openpyxl import load_workbook

    # read_only keeps memory flat on large books; data_only takes cached
    # formula results rather than the formula text itself.
    workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    raw = []
    for sheet in workbook.worksheets:
        lines = [f"Sheet: {sheet.title}"]
        for row in sheet.iter_rows(values_only=True):
            cells = [str(c) for c in row if c is not None and str(c).strip()]
            if cells:
                lines.append(" | ".join(cells))
        # A sheet with only its title line has no data worth indexing.
        raw.append("\n".join(lines) if len(lines) > 1 else "")
    workbook.close()

    pages = _as_pages(raw)
    if not pages:
        raise EmptyDocument("All worksheets are empty.")

    return ParsedFile(pages=pages, page_unit="worksheet")


def _parse_html(data: bytes) -> ParsedFile:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(_decode(data), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    pages = _as_pages(_paginate(text))
    if not pages:
        raise EmptyDocument("No readable text found in the HTML.")

    return ParsedFile(pages=pages, page_unit=f"~{WORDS_PER_SYNTHETIC_PAGE}-word block")


def _parse_csv(data: bytes) -> ParsedFile:
    text = _decode(data)
    reader = csv.reader(io.StringIO(text))

    try:
        rows = list(reader)
    except csv.Error as exc:
        raise EmptyDocument(f"Could not read the CSV: {exc}") from exc

    if not rows:
        raise EmptyDocument("The CSV is empty.")

    header, *body = rows
    header_line = " | ".join(str(c) for c in header)

    # Repeating the header on every page keeps each page independently
    # meaningful -- a retrieved chunk from page 7 still says what its
    # columns are.
    rows_per_page = 50
    raw = []
    for start in range(0, len(body), rows_per_page):
        lines = [header_line]
        lines += [" | ".join(str(c) for c in row) for row in body[start : start + rows_per_page]]
        raw.append("\n".join(lines))

    pages = _as_pages(raw) or _as_pages([header_line])
    return ParsedFile(pages=pages, page_unit=f"{rows_per_page} rows")


def _parse_text(data: bytes) -> ParsedFile:
    text = _decode(data)
    pages = _as_pages(_paginate(text))
    if not pages:
        raise EmptyDocument("The file contains no text.")

    unit = f"~{WORDS_PER_SYNTHETIC_PAGE}-word block" if len(pages) > 1 else "whole file"
    return ParsedFile(pages=pages, page_unit=unit)
