import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.logging_config import setup_logging
from app.api.routes import router
from app.retrieval.vector_store import close_pool

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up")
    yield
    logger.info("Application shutting down")
    close_pool()


app = FastAPI(title="RAG Chat", lifespan=lifespan)
app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok"}