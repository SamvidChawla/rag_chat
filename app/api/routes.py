import logging
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.ingestion.pipeline import ingest_document
from app.ingestion.embedder import embed_query
from app.retrieval.vector_store import search
from app.retrieval.reranker import rerank
from app.generation.llm_client import generate_answer

logger = logging.getLogger(__name__)
router = APIRouter()


class IngestResponse(BaseModel):
    source_name: str
    chunks_stored: int


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


async def _save_upload(file: UploadFile, dest_path: Path) -> None:
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    written = 0

    with open(dest_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1MB at a time
            written += len(chunk)
            if written > max_bytes:
                f.close()
                dest_path.unlink(missing_ok=True)
                logger.warning(
                    "Upload rejected: exceeds %dMB (file=%s)",
                    settings.max_upload_size_mb, file.filename,
                )
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds {settings.max_upload_size_mb}MB limit",
                )
            f.write(chunk)


@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(file: UploadFile = File(...)):
    temp_path = Path(tempfile.gettempdir()) / f"{uuid4()}_{file.filename}"
    logger.info("Ingest request received: %s", file.filename)

    try:
        await _save_upload(file, temp_path)
        result = ingest_document(str(temp_path))
        logger.info("Ingest succeeded: %s", file.filename)
        return result
    except HTTPException:
        raise
    except Exception:
        logger.error("Ingest failed: %s", file.filename, exc_info=True)
        raise HTTPException(status_code=500, detail="Ingestion failed")
    finally:
        temp_path.unlink(missing_ok=True)


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    logger.info("Query received (len=%d)", len(request.query))

    try:
        query_embedding = embed_query(request.query)
        chunks = search(query_embedding)
        chunks = rerank(request.query, chunks)
        answer = generate_answer(request.query, chunks)

        sources = list({c["source_name"] for c in chunks})
        logger.info("Query answered (answer_len=%d, sources=%d)", len(answer), len(sources))

        return QueryResponse(answer=answer, sources=sources)
    except Exception:
        logger.error("Query failed: %s", request.query, exc_info=True)
        raise HTTPException(status_code=500, detail="Query failed")