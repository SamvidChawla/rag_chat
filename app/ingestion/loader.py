import logging
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from docx import Document as DocxDocument

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def load_document(file_path: str) -> dict[str, Any]:
    """
    Load a document's text content and basic metadata.

    Returns:
        {"text": str, "metadata": {"source_name": str, "extension": str, "page_count": int | None}}
    """
    path = Path(file_path)

    if not path.exists():
        logger.error("File not found: %s", file_path)
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        logger.error("Unsupported file extension: %s (%s)", ext, file_path)
        raise ValueError(f"Unsupported file extension: {ext}")

    try:
        if ext == ".pdf":
            text, page_count = _load_pdf(path)
        elif ext == ".docx":
            text, page_count = _load_docx(path)
        else:  # .txt, .md
            text, page_count = _load_plain_text(path)
    except Exception:
        logger.error("Failed to load document: %s", file_path, exc_info=True)
        raise

    if not text.strip():
        logger.error("Loaded document is empty: %s", file_path)
        raise ValueError(f"Document has no extractable text: {file_path}")

    logger.info(
        "Loaded document: %s (%d chars, ext=%s)", path.name, len(text), ext
    )

    return {
        "text": text,
        "metadata": {
            "source_name": path.name,
            "extension": ext,
            "page_count": page_count,
        },
    }


def _load_pdf(path: Path) -> tuple[str, int]:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages), len(reader.pages)


def _load_docx(path: Path) -> tuple[str, None]:
    doc = DocxDocument(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs), None


def _load_plain_text(path: Path) -> tuple[str, None]:
    return path.read_text(encoding="utf-8"), None