"""POST /ask. Run: uv run --group models uvicorn rag.api:app"""
import logging
import os
from contextlib import asynccontextmanager
from datetime import date
from typing import Literal

import anthropic
import psycopg
from google.genai import errors as gemini_errors
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool, PoolTimeout
from pydantic import BaseModel, Field, field_validator

from versioning import parse_as_of

from . import store
from .answer import answer
from .models import LLM, Embedder, LLMError
from .pipeline import retrieve

log = logging.getLogger(__name__)


class Question(BaseModel):
    question: str = Field(max_length=2000, pattern=r"\S")
    mode: Literal["all", "current", "aware", "lexical"] = "aware"
    as_of: date | None = None  # overrides any date in the question

    @field_validator("question")
    @classmethod
    def no_nul(cls, value: str) -> str:
        if "\x00" in value:  # Postgres text can't hold NUL
            raise ValueError("question contains a NUL character")
        return value


class Citation(BaseModel):
    document_id: str
    section: str


class Source(BaseModel):
    document_id: str
    version: int
    section: str
    effective_from: date
    effective_to: date | None
    score: float


class Answer(BaseModel):
    answer: str
    citations: list[Citation]
    as_of: date | None  # the date retrieval filtered on; null for modes all and current, which ignore it
    sources: list[Source]


class Services:
    """Loaded once: a connection pool that replaces broken connections, the embedder and the model client."""

    def __init__(self) -> None:
        self.pool = ConnectionPool(os.environ.get("DATABASE_URL", store.DEFAULT_URL), kwargs={"autocommit": True},
                                   configure=register_vector, check=ConnectionPool.check_connection, timeout=10,
                                   open=True)
        self.embedder, self.llm = Embedder(), LLM()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.services = Services()
    yield
    app.state.services.pool.close()


app = FastAPI(title="Version-aware RAG", lifespan=lifespan)


def services(request: Request) -> Services:
    return request.app.state.services


def search(svc: Services, question: str, mode: str, as_of: date) -> list[dict]:
    """Retrieve on a pooled connection, returned to the pool before the model is called."""
    with svc.pool.connection() as conn:
        built = store.meta(conn)
        if built is None or built["embedder"] != svc.embedder.name:
            raise HTTPException(503, f"no index built with {svc.embedder.name}; run python -m rag.pipeline index")
        return retrieve(conn, svc.embedder, question, mode, as_of)


@app.exception_handler(LLMError)
@app.exception_handler(anthropic.APIError)
@app.exception_handler(gemini_errors.APIError)
async def model_failed(request: Request, exc: Exception) -> JSONResponse:
    log.error("model call failed: %s", exc, exc_info=exc)
    return JSONResponse({"detail": "the language model failed to answer"}, status_code=502)


@app.exception_handler(psycopg.OperationalError)
@app.exception_handler(PoolTimeout)
async def database_down(request: Request, exc: Exception) -> JSONResponse:
    log.error("database unavailable: %s", exc, exc_info=exc)
    return JSONResponse({"detail": "database unavailable"}, status_code=503)


@app.post("/ask")
def post_ask(body: Question, svc: Services = Depends(services)) -> Answer:
    as_of = body.as_of or parse_as_of(body.question, today=date.today())
    applied = as_of if body.mode in ("aware", "lexical") else None
    chunks = search(svc, body.question, body.mode, as_of)
    result = answer(svc.llm, body.question, chunks, applied)
    return Answer(
        answer=result["text"],
        citations=[Citation(document_id=d, section=s) for d, s in result["citations"]],
        as_of=applied,
        sources=[Source(**{k: c[k] for k in Source.model_fields}) for c in chunks],
    )
