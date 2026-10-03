"""Pure text helpers for the Streamlit console, kept apart from app.py so they
can be imported (and tested) without starting the UI."""

from __future__ import annotations

import re

_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_REF_IMAGE = re.compile(r"!\[([^\]]*)\]\[[^\]]*\]")
_MD_REF_DEFINITION = re.compile(r"^ {0,3}\[[^\]]+\]:\s*\S+.*$", re.MULTILINE)
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")


def safe_answer(text: str) -> str:
    """Model output made safe to render as markdown.

    A planted instruction inside a document could make the model emit an image
    whose URL carries retrieved text; the browser would fetch it -- sending the
    text to that server -- the moment the answer renders. Images are dropped
    (alt text kept) and links become their plain text. Citation markers like
    [1] have no URL, so they are untouched.
    """
    text = _MD_IMAGE.sub(r"\1", text)
    text = _MD_REF_IMAGE.sub(r"\1", text)
    text = _MD_REF_DEFINITION.sub("", text)
    return _MD_LINK.sub(r"\1", text)


def slugify(filename: str) -> str:
    """Derive a stable external_id from a filename, extension included.

    Stable matters: re-uploading the same file must produce the same id so the
    API treats it as an update and skips unchanged pages, rather than creating
    a duplicate document. The extension is kept so report.pdf and report.docx
    stay two documents instead of one silently replacing the other.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", filename.lower()).strip("-")
    return slug or "document"
