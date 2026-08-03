import argparse
import logging
import sys
from pathlib import Path

from app.logging_config import setup_logging
from app.ingestion.loader import SUPPORTED_EXTENSIONS
from app.ingestion.pipeline import ingest_document

logger = logging.getLogger(__name__)


def collect_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return [
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    raise FileNotFoundError(f"Path does not exist: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector store.")
    parser.add_argument("path", help="File or directory to ingest")
    args = parser.parse_args()

    setup_logging()

    files = collect_files(Path(args.path))
    if not files:
        logger.warning("No supported files found at: %s", args.path)
        sys.exit(0)

    logger.info("Found %d file(s) to ingest", len(files))

    succeeded, failed = 0, 0
    for file_path in files:
        try:
            ingest_document(str(file_path))
            succeeded += 1
        except Exception:
            logger.error("Failed to ingest: %s", file_path, exc_info=True)
            failed += 1

    logger.info("Ingestion run complete: %d succeeded, %d failed", succeeded, failed)


if __name__ == "__main__":
    main()