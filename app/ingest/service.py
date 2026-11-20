"""Atomic policy ingestion.

1. chunk  2. write NEW collection  3. verify count  4. flip registry pointer in one txn.
Any failure before step 4 drops the new collection and leaves the old version live.
"""

from app.store.registry import PolicyVersion, Registry
from app.store.vector import VectorStore


class IngestError(Exception):
    pass


def ingest_policy(
    company_id: str, doc_title: str, markdown: str, registry: Registry, store: VectorStore
) -> PolicyVersion:
    raise NotImplementedError
