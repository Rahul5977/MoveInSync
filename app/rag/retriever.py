"""Retrieval restricted to the caller's company's ACTIVE collection.

The collection comes from the registry pointer for `company_id` (taken from the JWT). There is
no parameter through which a client can name a collection, company or version.
"""

import logging

from app.schemas import RetrievedChunk
from app.store.registry import Registry
from app.store.vector import VectorStore

log = logging.getLogger(__name__)


def retrieve(
    company_id: str, question: str, registry: Registry, store: VectorStore, k: int
) -> tuple[list[RetrievedChunk], int | None]:
    """Return (chunks, active_version). Empty list + None if the company has no live policy."""
    active = registry.active(company_id)
    if active is None:
        return [], None

    chunks = store.query(active.collection_name, question, k)

    # Defence in depth: physical isolation should make this a no-op. If it ever filters
    # something, that is a bug worth shouting about, not silently serving.
    safe = [c for c in chunks if c.company_id == company_id and c.version == active.version]
    if len(safe) != len(chunks):
        log.error("Isolation filter dropped %d foreign chunks", len(chunks) - len(safe))
    return safe, active.version
