"""SQLite registry: the single source of truth for which policy version is live.

Tables
- active_version   one row per company -> the ONLY pointer retrieval reads.
- policy_versions  audit history of every ingest attempt (pending/active/retired/failed).

Activation updates both tables in ONE transaction (BEGIN IMMEDIATE), so a reader sees either
the old pointer or the new one, never a mix. Activation refuses to move a company backwards
or sideways (new version must be greater than the active one): once v2 is live, v1 is dead.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS policy_versions (
    company_id      TEXT    NOT NULL,
    version         INTEGER NOT NULL,
    collection_name TEXT    NOT NULL,
    doc_title       TEXT    NOT NULL,
    status          TEXT    NOT NULL CHECK (status IN ('pending','active','retired','failed')),
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT    NOT NULL,
    PRIMARY KEY (company_id, version)
);
CREATE TABLE IF NOT EXISTS active_version (
    company_id      TEXT    PRIMARY KEY,
    version         INTEGER NOT NULL,
    collection_name TEXT    NOT NULL,
    activated_at    TEXT    NOT NULL
);
"""


class VersionConflict(Exception):
    """Tried to activate a version that is not newer than the active one."""


@dataclass(frozen=True)
class ActivePolicy:
    company_id: str
    version: int
    collection_name: str


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Registry:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path, isolation_level=None, timeout=5.0)
        try:
            yield conn
        finally:
            conn.close()

    def init_schema(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def active(self, company_id: str) -> ActivePolicy | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT company_id, version, collection_name FROM active_version"
                " WHERE company_id = ?",
                (company_id,),
            ).fetchone()
        return ActivePolicy(*row) if row else None

    def record_pending(
        self, company_id: str, version: int, collection_name: str, doc_title: str
    ) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO policy_versions"
                " (company_id, version, collection_name, doc_title, status, created_at)"
                " VALUES (?, ?, ?, ?, 'pending', ?)"
                " ON CONFLICT (company_id, version) DO UPDATE SET"
                "   status = 'pending', doc_title = excluded.doc_title,"
                "   created_at = excluded.created_at",
                (company_id, version, collection_name, doc_title, _now()),
            )

    def mark_failed(self, company_id: str, version: int) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE policy_versions SET status = 'failed'"
                " WHERE company_id = ? AND version = ? AND status = 'pending'",
                (company_id, version),
            )

    def activate(
        self, company_id: str, version: int, collection_name: str, chunk_count: int
    ) -> None:
        """Flip the pointer to `version` atomically. Raises VersionConflict if not newer."""
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            try:
                row = c.execute(
                    "SELECT version FROM active_version WHERE company_id = ?", (company_id,)
                ).fetchone()
                if row is not None and row[0] >= version:
                    raise VersionConflict(
                        f"{company_id}: v{version} is not newer than active v{row[0]}"
                    )
                c.execute(
                    "UPDATE policy_versions SET status = 'retired'"
                    " WHERE company_id = ? AND status = 'active'",
                    (company_id,),
                )
                c.execute(
                    "UPDATE policy_versions SET status = 'active', chunk_count = ?"
                    " WHERE company_id = ? AND version = ?",
                    (chunk_count, company_id, version),
                )
                c.execute(
                    "INSERT INTO active_version"
                    " (company_id, version, collection_name, activated_at)"
                    " VALUES (?, ?, ?, ?)"
                    " ON CONFLICT (company_id) DO UPDATE SET version = excluded.version,"
                    "   collection_name = excluded.collection_name,"
                    "   activated_at = excluded.activated_at",
                    (company_id, version, collection_name, _now()),
                )
                c.execute("COMMIT")
            except BaseException:
                c.execute("ROLLBACK")
                raise

    def unfinished(self) -> list[tuple[str, int, str, str]]:
        """(company_id, version, collection_name, status) for every pending/failed attempt."""
        with self._conn() as c:
            return c.execute(
                "SELECT company_id, version, collection_name, status FROM policy_versions"
                " WHERE status IN ('pending', 'failed') ORDER BY company_id, version"
            ).fetchall()

    def history(self, company_id: str) -> list[tuple[int, str, int]]:
        with self._conn() as c:
            return c.execute(
                "SELECT version, status, chunk_count FROM policy_versions"
                " WHERE company_id = ? ORDER BY version",
                (company_id,),
            ).fetchall()
