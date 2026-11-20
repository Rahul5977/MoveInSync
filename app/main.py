"""FastAPI entrypoint."""

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status

from app.auth import current_principal, require_admin
from app.schemas import AskRequest, AskResponse, Principal

app = FastAPI(title="Commute Policy Assistant", version="0.1.0")


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


@app.post("/admin/ingest")
def ingest(principal: Annotated[Principal, Depends(require_admin)]) -> dict:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, "ingestion not implemented yet")
