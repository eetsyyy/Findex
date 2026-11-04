"""Findex HTTP API and server-rendered search UI (Lab 07)."""
from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.responses import Response

from findex.models import Index
from findex.numpy_ranking import cached_numpy_searcher
from findex.ranking import BM25, RankedSearcher, TfIdf
from findex.ranking import SearchResult as RankedHit
from findex.store import load

logger = logging.getLogger("findex.web")
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


class Settings(BaseSettings):
    """Deployment configuration, read from the environment."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    index_path: Path = Path("index.bin")
    corpus_path: Path | None = None
    log_level: str = "INFO"
    port: int = Field(default=8000, ge=1, le=65535)


class ResultModel(BaseModel):
    doc_id: int
    title: str
    path: str
    score: float
    snippet: str = ""


class SearchResponse(BaseModel):
    q: str
    k: int
    page: int
    scorer: Literal["bm25", "tfidf"]
    engine: Literal["legacy", "numpy"] = "numpy"
    total: int
    pages: int
    results: list[ResultModel]


class DocumentResponse(BaseModel):
    doc_id: int
    title: str
    path: str
    text: str | None


class StatsResponse(BaseModel):
    documents: int
    terms: int
    avg_document_length: float


def get_index(request: Request) -> Index:
    """Replace this dependency in TestClient tests without disk access."""
    index: Index | None = getattr(request.app.state, "index", None)
    if index is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    return index


def get_corpus(request: Request) -> Path | None:
    return getattr(request.app.state, "corpus_path", None)


def _read_text(index: Index, doc_id: int, corpus: Path | None) -> str | None:
    if corpus is None:
        return None
    base = corpus.resolve()
    candidate = (base / index.doc_meta[doc_id].path).resolve()
    if not candidate.is_relative_to(base) or not candidate.is_file():
        return None
    return candidate.read_text(encoding="utf-8", errors="replace")


def _calculate(
    index: Index,
    q: str,
    k: int,
    scorer: Literal["bm25", "tfidf"],
    page: int,
    corpus: Path | None,
    engine: Literal["legacy", "numpy"] = "numpy",
) -> SearchResponse:
    searcher = RankedSearcher(index)
    try:
        total = len(searcher.matching_ids(q))
        limit = min(total, k * page)
        score_fn = BM25() if scorer == "bm25" else TfIdf()
        found: list[RankedHit] = (
            cached_numpy_searcher(index).search(q, scorer=scorer, k=limit, corpus=corpus)
            if engine == "numpy" else searcher.search(q, scorer=score_fn, k=limit, corpus=corpus)
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid search syntax") from exc
    results = [ResultModel(doc_id=h.doc_id, title=h.title, path=h.path,
                           score=h.score, snippet=h.snippet) for h in found[(page - 1) * k:]]
    return SearchResponse(q=q, k=k, page=page, scorer=scorer, engine=engine, total=total,
                          pages=(total + k - 1) // k, results=results)


def _highlight(text: str, query: str) -> str:
    """Escape every source character; mark only literal matched words."""
    words = [w for w in re.findall(r"\w+", query) if w.lower() not in {"and", "or", "not"}]
    if not words:
        return escape(text)
    pattern = re.compile("(" + "|".join(re.escape(w) for w in sorted(set(words), key=len, reverse=True)) + ")", re.IGNORECASE)
    return "".join(f"<mark>{escape(part)}</mark>" if pattern.fullmatch(part) else escape(part)
                   for part in pattern.split(text.replace("**", "")))


@asynccontextmanager
async def lifespan(application: FastAPI):
    settings: Settings = application.state.settings
    logging.basicConfig(level=settings.log_level.upper())
    application.state.index = None
    application.state.corpus_path = settings.corpus_path
    try:
        # The index is our own trusted artifact; never load an untrusted pickle.
        application.state.index, _ = load(settings.index_path)
        logger.info("Index loaded: %s", settings.index_path)
    except (OSError, ValueError, EOFError, ImportError) as exc:
        logger.error("Index unavailable at startup: %s", exc)
    yield
    application.state.index = None


def create_app(settings: Settings | None = None) -> FastAPI:
    application = FastAPI(title="Findex Search", version="1.0.0", lifespan=lifespan)
    application.state.settings = settings or Settings()
    application.state.index = None
    application.state.corpus_path = application.state.settings.corpus_path

    @application.middleware("http")
    async def request_tracking(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))[:128]
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled request error request_id=%s path=%s", request_id, request.url.path)
            response = JSONResponse(status_code=500, content={"detail": "Internal server error", "request_id": request_id})
        response.headers["X-Request-ID"] = request_id
        return response

    @application.get("/health")
    def health(request: Request) -> dict[str, str]:
        if getattr(request.app.state, "index", None) is None:
            raise HTTPException(status_code=503, detail="Index not loaded")
        return {"status": "ok"}

    @application.get("/stats", response_model=StatsResponse)
    def stats(index: Annotated[Index, Depends(get_index)]) -> StatsResponse:
        return StatsResponse(documents=index.num_docs, terms=len(index),
                             avg_document_length=index.avg_doc_length)

    @application.get("/search", response_model=SearchResponse)
    def search(
        q: Annotated[str, Query(min_length=1, max_length=200)],
        index: Annotated[Index, Depends(get_index)],
        corpus: Annotated[Path | None, Depends(get_corpus)],
        k: Annotated[int, Query(ge=1, le=100)] = 10,
        scorer: Literal["bm25", "tfidf"] = "bm25",
        engine: Literal["legacy", "numpy"] = "numpy",
        page: Annotated[int, Query(ge=1, le=100)] = 1,
    ) -> SearchResponse:
        # Plain def routes run in FastAPI's thread pool: CPU scoring never blocks the event loop.
        return _calculate(index, q, k, scorer, page, corpus, engine)

    @application.get("/search-async", response_model=SearchResponse, include_in_schema=False)
    async def search_async_benchmark(
        q: Annotated[str, Query(min_length=1, max_length=200)],
        index: Annotated[Index, Depends(get_index)],
        corpus: Annotated[Path | None, Depends(get_corpus)],
        k: Annotated[int, Query(ge=1, le=100)] = 10,
        scorer: Literal["bm25", "tfidf"] = "bm25",
        engine: Literal["legacy", "numpy"] = "numpy",
        page: Annotated[int, Query(ge=1, le=100)] = 1,
    ) -> SearchResponse:
        # Intentionally synchronous CPU code in async route, exclusively for benchmark comparison.
        return _calculate(index, q, k, scorer, page, corpus, engine)

    @application.get("/docs/{doc_id}", response_model=DocumentResponse)
    def document(doc_id: int, index: Annotated[Index, Depends(get_index)],
                 corpus: Annotated[Path | None, Depends(get_corpus)]) -> DocumentResponse:
        metadata = index.doc_meta.get(doc_id)
        if metadata is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return DocumentResponse(doc_id=doc_id, title=metadata.title, path=metadata.path,
                                text=_read_text(index, doc_id, corpus))

    @application.get("/", response_class=HTMLResponse)
    def home(request: Request, q: str = "", k: int = 10, page: int = 1, engine: Literal["legacy", "numpy"] = "numpy"):
        index: Index | None = getattr(request.app.state, "index", None)
        data: SearchResponse | None = None
        error: str | None = None
        if index is None:
            error = "Індекс ще не завантажений. Перевірте INDEX_PATH."
        elif q.strip():
            try:
                if not 1 <= k <= 100 or not 1 <= page <= 100 or len(q) > 200:
                    raise ValueError("Invalid search parameters")
                data = _calculate(index, q, k, "bm25", page, get_corpus(request), engine)
            except (HTTPException, ValueError):
                error = "Некоректний пошуковий запит."
        return TEMPLATES.TemplateResponse(request=request, name="index.html",
                                          context={"q": q, "engine": engine, "data": data, "error": error,
                                                   "highlight": _highlight})

    @application.get("/document/{doc_id}", response_class=HTMLResponse)
    def document_page(request: Request, doc_id: int,
                      index: Annotated[Index, Depends(get_index)],
                      corpus: Annotated[Path | None, Depends(get_corpus)]):
        metadata = index.doc_meta.get(doc_id)
        if metadata is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return TEMPLATES.TemplateResponse(request=request, name="document.html",
                                          context={"doc_id": doc_id, "title": metadata.title,
                                                   "path": metadata.path, "text": _read_text(index, doc_id, corpus)})
    return application


app = create_app()
