"""FastAPI entrypoint. `create_app()` builds the app; tests pass their own settings/store."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel

from app.auth import current_principal, require_admin
from app.config import Settings, get_settings
from app.ingest.service import (
    IngestFailed,
    InvalidDocument,
    ingest_policy,
    reconcile,
    seed_policies,
)
from app.schemas import AskRequest, AskResponse, Principal
from app.store.registry import Registry, VersionConflict
from app.store.vector import UnknownCompany, VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


async def only_file_field(request: Request) -> None:
    """Reject any multipart field other than `file` (e.g. a `company_id` override attempt).

    The company is always the admin's own, from the JWT (D21, D29).
    """
    extra = set((await request.form()).keys()) - {"file"}
    if extra:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"Unexpected form fields: {sorted(extra)}"
        )


class IngestResponse(BaseModel):
    company_id: str
    version: int
    collection: str
    chunk_count: int


def create_app(
    settings: Settings | None = None,
    store: VectorStore | None = None,
    registry: Registry | None = None,
) -> FastAPI:
    s = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        reg = registry or Registry(s.registry_path)
        reg.init_schema()
        vs = store or VectorStore(s.chroma_dir)
        app.state.settings, app.state.registry, app.state.store = s, reg, vs
        app.state.reconcile_outcomes = reconcile(reg, vs)  # D16: before seeding
        app.state.seed_outcomes = seed_policies(reg, vs, s)
        yield

    app = FastAPI(title="Commute Policy Assistant", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(
        req: AskRequest,
        principal: Annotated[Principal, Depends(current_principal)],
    ) -> AskResponse:
        # company_id comes from `principal` (JWT) only — never from `req`.
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, "answerer not implemented yet")

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
