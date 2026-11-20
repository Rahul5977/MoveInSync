"""FastAPI entrypoint. `create_app()` builds the app; tests pass their own settings/store."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.auth import current_principal, require_admin
from app.config import Settings, get_settings
from app.embeddings import make_embedder
from app.errors import not_allowed, validation_message
from app.ingest.service import (
    IngestFailed,
    InvalidDocument,
    ingest_policy,
    reconcile,
    seed_policies,
)
from app.llm.base import LLMClient
from app.llm.factory import make_llm
from app.rag.answerer import answer_question
from app.rag.bm25 import Bm25Cache
from app.schemas import AnswerStatus, AskRequest, AskResponse, Principal
from app.store.registry import Registry, VersionConflict
from app.store.vector import UnknownCompany, VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")


async def only_file_field(request: Request) -> None:
    """Reject any multipart field other than `file` (e.g. a `company_id` override attempt).

    The company is always the admin's own, from the JWT (D21, D29).
    """
    extra = set((await request.form()).keys()) - {"file"}
    if extra:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, not_allowed(extra))


class IngestResponse(BaseModel):
    company_id: str
    version: int
    collection: str
    chunk_count: int


def create_app(
    settings: Settings | None = None,
    store: VectorStore | None = None,
    registry: Registry | None = None,
    llm: LLMClient | None = None,
) -> FastAPI:
    s = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        reg = registry or Registry(s.registry_path)
        reg.init_schema()
        vs = store or VectorStore(s.chroma_dir, make_embedder(s.embedding_model, s.bge_model_dir))
        app.state.settings, app.state.registry, app.state.store = s, reg, vs
        app.state.llm = llm or make_llm(s)
        if not app.state.llm.ready:  # D38: one warning, then the app still starts
            log.warning(
                "LLM provider %r is not ready (missing GROQ_API_KEY?): /ask will return try_again",
                app.state.llm.provider,
            )
        app.state.reconcile_outcomes = reconcile(reg, vs)  # D16: before seeding
        for company_id in s.companies:  # D59: refuse to search vectors from another model
            active = reg.active(company_id)
            if active:
                vs.check_model(active.collection_name)
        app.state.seed_outcomes = seed_policies(reg, vs, s)
        app.state.bm25 = Bm25Cache()  # D56: warm one index per active collection
        for company_id in s.companies:
            active = reg.active(company_id)
            if active:
                app.state.bm25.build(vs, active.collection_name)
        yield

    app = FastAPI(title="Commute Policy Assistant", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def plain_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        """422 with one plain sentence; never Pydantic's error list or the input (D42)."""
        return JSONResponse(status_code=422, content={"detail": validation_message(exc.errors())})

    @app.get("/health")
    def health(request: Request) -> dict:
        # D35 + D38: no tenant data; just whether answers are possible right now.
        llm = request.app.state.llm
        return {"status": "ok", "llm": {"provider": llm.provider, "ready": bool(llm.ready)}}

    @app.post("/ask", response_model=AskResponse, responses={503: {"model": AskResponse}})
    def ask(
        req: AskRequest,
        request: Request,
        principal: Annotated[Principal, Depends(current_principal)],
    ):
        # company_id comes from `principal` (JWT) only — never from `req`.
        st = request.app.state
        resp, trace = answer_question(
            principal, req.question, st.registry, st.store, st.llm, s, bm25=st.bm25
        )
        log.info(
            "ask company=%s status=%s gate=%s best=%s sent=%s calls=%d ms=%d",
            principal.company_id,
            resp.status,
            trace.gate,
            None if trace.best_score is None else round(trace.best_score, 3),
            trace.sent_ids,
            resp.meta.llm_calls,
            resp.meta.latency_ms,
        )
        if resp.status == AnswerStatus.TRY_AGAIN:
            return JSONResponse(status_code=503, content=resp.model_dump(mode="json"))
        return resp

    @app.post(
        "/admin/ingest",
        status_code=201,
        response_model=IngestResponse,
        dependencies=[Depends(only_file_field)],
    )
    def ingest(
        file: Annotated[UploadFile, File(description="Policy document (.md, UTF-8)")],
        request: Request,
        principal: Annotated[Principal, Depends(require_admin)],
    ) -> IngestResponse:
        raw = file.file.read(s.max_document_chars * 4 + 1)  # UTF-8 is at most 4 bytes/char
        try:
            document = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, "Policy file must be UTF-8 text"
            ) from exc
        st = request.app.state
        try:
            res = ingest_policy(principal.company_id, document, st.registry, st.store, s)
        except (InvalidDocument, UnknownCompany) as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
        except VersionConflict as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        except IngestFailed as exc:
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc)) from exc
        st.bm25.build(st.store, res.collection_name)  # D56: rebuilt on activation
        return IngestResponse(
            company_id=res.company_id,
            version=res.version,
            collection=res.collection_name,
            chunk_count=res.chunk_count,
        )

    @app.get("/admin/policy")
    def active_policy(
        request: Request, principal: Annotated[Principal, Depends(require_admin)]
    ) -> dict:
        reg: Registry = request.app.state.registry
        active = reg.active(principal.company_id)
        return {
            "company_id": principal.company_id,
            "active_version": active.version if active else None,
            "history": [
                {"version": v, "status": st, "chunk_count": n}
                for v, st, n in reg.history(principal.company_id)
            ],
        }

    return app


app = create_app()
