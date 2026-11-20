"""SQLite registry: the single source of truth for which policy version is live.

Invariant: at most one `active` row per company (enforced by a partial unique index).
Activation retires the old version and activates the new one in ONE transaction.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PolicyVersion:
    company_id: str
    version: int
    collection_name: str
    doc_title: str
    status: str
    chunk_count: int


class Registry:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def init_schema(self) -> None:
        raise NotImplementedError  # TODO(D3): create table + partial unique index

    def next_version(self, company_id: str) -> int:
        raise NotImplementedError

    def create_pending(self, pv: PolicyVersion) -> None:
        raise NotImplementedError

    def activate(self, company_id: str, version: int) -> None:
        """Retire current active + activate `version` atomically."""
        raise NotImplementedError

    def mark_failed(self, company_id: str, version: int) -> None:
        raise NotImplementedError

    def active(self, company_id: str) -> PolicyVersion | None:
        raise NotImplementedError
