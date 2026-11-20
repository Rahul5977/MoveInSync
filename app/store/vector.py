"""Thin wrapper over Chroma. One collection per (company, version)."""

from pathlib import Path

from app.schemas import RetrievedChunk


def collection_name(company_id: str, version: int) -> str:
    return f"{company_id}__v{version}"


class VectorStore:
    def __init__(self, persist_dir: Path) -> None:
        self._persist_dir = persist_dir

    def write_chunks(self, name: str, chunks: list[dict]) -> int:
        """Create collection `name`, add chunks, return verified count."""
        raise NotImplementedError

    def drop(self, name: str) -> None:
        raise NotImplementedError

    def query(self, name: str, text: str, k: int) -> list[RetrievedChunk]:
        raise NotImplementedError
