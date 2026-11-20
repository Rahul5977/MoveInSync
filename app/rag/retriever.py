"""Retrieval restricted to the caller's company's ACTIVE collection."""

from app.schemas import RetrievedChunk
from app.store.registry import Registry
from app.store.vector import VectorStore


def retrieve(
    company_id: str, question: str, registry: Registry, store: VectorStore, k: int
) -> tuple[list[RetrievedChunk], int | None]:
    """Return (chunks, active_version). company_id must come from the JWT."""
    raise NotImplementedError
