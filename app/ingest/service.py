"""Atomic policy ingestion + idempotent startup seeding.

ingest_policy():
  1. validate company (allow-list) and size
  2. parse + chunk                      -> InvalidDocument on bad format / wrong company
  3. refuse if version <= active        -> VersionConflict (no rollback, no re-ingest)
  4. write ALL chunks to a NEW collection `<company>_v<n>`, verify count
  5. flip the pointer in ONE SQLite transaction (Registry.activate)
  Any failure before step 5 leaves the pointer untouched, so the old version keeps answering.
  The half-written collection is dropped best-effort; if that fails it is an unreachable orphan.

reconcile():
  Runs at startup BEFORE seeding. A process crash mid-ingest never reaches the except block,
  so it leaves a `pending` row and a half-written collection. Mark such rows `failed` and drop
  the collections of all pending/failed attempts (never the live one).

seed_policies():
  For each company folder in policies/seed/, ingest it ONLY if the company has no active
  version yet. A restart therefore never rolls a company back (e.g. Nimbus v2 -> v1).
"""

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass

from app.config import Settings
from app.ingest.chunker import ChunkingError, chunk_policy, parse_policy
from app.store.registry import Registry, VersionConflict
from app.store.vector import UnknownCompany, VectorStore, collection_name

log = logging.getLogger(__name__)

# One ingest per company at a time (single-process server). Prevents two concurrent ingests of
# the same version from overwriting each other's collection.
_company_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)


class InvalidDocument(ValueError):
    pass


class IngestFailed(RuntimeError):
    pass


@dataclass(frozen=True)
class IngestResult:
    company_id: str
    version: int
    collection_name: str
    chunk_count: int


def ingest_policy(
    company_id: str,
    markdown: str,
    registry: Registry,
    store: VectorStore,
    settings: Settings,
) -> IngestResult:
    if company_id not in settings.companies:
        raise UnknownCompany(f"Unknown company_id {company_id!r}")
    if len(markdown) > settings.max_document_chars:
        raise InvalidDocument(f"Document exceeds {settings.max_document_chars} characters")

    try:
        policy = parse_policy(markdown)
    except ChunkingError as exc:
        raise InvalidDocument(str(exc)) from exc

    expected_name = settings.companies[company_id]
    if policy.company_name != expected_name:
        raise InvalidDocument(
            f"Document is for {policy.company_name!r}, but this company is {expected_name!r}"
        )

    chunks = chunk_policy(company_id, policy, settings.chunk_max_words)
    name = collection_name(company_id, policy.version, settings.companies)

    with _company_locks[company_id]:
        active = registry.active(company_id)
        if active is not None and policy.version <= active.version:
            raise VersionConflict(
                f"{company_id}: v{policy.version} is not newer than active v{active.version}"
            )

        registry.record_pending(company_id, policy.version, name, policy.doc_title)
        try:
            count = store.write_chunks(name, chunks, settings.ingest_batch_size)
            registry.activate(company_id, policy.version, name, count)
        except BaseException as exc:
            registry.mark_failed(company_id, policy.version)
            try:
                store.drop(name)
            except Exception:  # cleanup is best-effort; orphan is unreachable anyway
                log.exception("Could not drop orphan collection %s", name)
            if isinstance(exc, VersionConflict):
                raise
            raise IngestFailed(
                f"Ingestion of {name} failed; previous version still active"
            ) from exc

    log.info("Activated %s (%d chunks)", name, count)
    return IngestResult(company_id, policy.version, name, count)


def reconcile(registry: Registry, store: VectorStore) -> dict[str, str]:
    """Clean up after crashed or failed ingests. Returns {collection: outcome}."""
    outcomes: dict[str, str] = {}
    for company_id, version, name, status in registry.unfinished():
        active = registry.active(company_id)
        if active is not None and active.collection_name == name:
            outcomes[name] = "kept: is the live collection"  # cannot happen; guard anyway
            continue
        if status == "pending":
            registry.mark_failed(company_id, version)
        if store.exists(name):
            store.drop(name)
            outcomes[name] = f"{status} -> failed, collection dropped"
        elif status == "pending":
            outcomes[name] = "pending -> failed"
    if outcomes:
        log.warning("Reconciled unfinished ingests: %s", outcomes)
    return outcomes


def seed_policies(registry: Registry, store: VectorStore, settings: Settings) -> dict[str, str]:
    """Ingest policies/seed/<company>/*.md for companies with no active version.

    Returns {company_id: outcome} for logging/tests. One company failing does not block others.
    """
    outcomes: dict[str, str] = {}
    if not settings.seed_dir.is_dir():
        log.warning("Seed dir %s not found", settings.seed_dir)
        return outcomes

    for company_dir in sorted(p for p in settings.seed_dir.iterdir() if p.is_dir()):
        company_id = company_dir.name
        if company_id not in settings.companies:
            outcomes[company_id] = "skipped: unknown company"
            continue
        if registry.active(company_id) is not None:
            outcomes[company_id] = "skipped: already active"
            continue
        files = sorted(company_dir.glob("*.md"))
        if len(files) != 1:
            outcomes[company_id] = f"error: expected 1 seed file, found {len(files)}"
            continue
        try:
            res = ingest_policy(company_id, files[0].read_text(), registry, store, settings)
            outcomes[company_id] = f"ingested v{res.version}"
        except Exception as exc:
            log.exception("Seeding %s failed", company_id)
            outcomes[company_id] = f"error: {exc}"
    log.info("Seed outcomes: %s", outcomes)
    return outcomes
