"""Thin wrapper over Chroma. One collection per (company, version), e.g. `saffron_v1`.

Collection names are only ever built by `collection_name()` from a server-side company_id
that is validated against the configured allow-list. Nothing user-supplied reaches it.
"""

import re
from collections.abc import Iterable
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from app.embeddings import DEFAULT_MODEL, Embedder, MiniLmEmbedder
from app.ingest.chunker import Chunk
from app.schemas import RetrievedChunk

_COMPANY_ID_RE = re.compile(r"^[a-z][a-z0-9]{1,30}$")


class UnknownCompany(ValueError):
    pass


EMBEDDING_CHANGED_MSG = "embedding model changed; reset with docker compose down -v"


class EmbeddingModelMismatch(RuntimeError):
    """A collection was embedded with a different model than the one configured (D59)."""

    def __init__(self, name: str, stored: str, configured: str) -> None:
        super().__init__(
            f"{EMBEDDING_CHANGED_MSG} ({name} was embedded with {stored!r}, "
            f"EMBEDDING_MODEL is {configured!r})"
        )


def collection_name(company_id: str, version: int, known_companies: Iterable[str]) -> str:
    if company_id not in set(known_companies) or not _COMPANY_ID_RE.match(company_id):
        raise UnknownCompany(f"Unknown company_id {company_id!r}")
    if version < 1:
        raise ValueError(f"Invalid version {version}")
    return f"{company_id}_v{version}"


class VectorStore:
    def __init__(self, persist_dir: Path, embedder: Embedder | None = None) -> None:
        # D59: we embed ourselves (documents vs queries differ for bge); Chroma only stores.
        self.embedder = embedder or MiniLmEmbedder()
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )

    def write_chunks(self, name: str, chunks: list[Chunk], batch_size: int) -> int:
        """(Re)create collection `name`, write all chunks in batches, verify the count.

        Any leftover collection with the same name (e.g. from an earlier failed attempt)
        is dropped first, so a retry always starts clean.
        """
        self.drop(name)
        col = self._client.create_collection(
            name,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedding_model": self.embedder.name},
        )
        for i in range(0, len(chunks), batch_size):
            self._add_batch(col, chunks[i : i + batch_size])
        written = col.count()
        if written != len(chunks):
            raise RuntimeError(f"{name}: wrote {written} chunks, expected {len(chunks)}")
        return written

    def _add_batch(self, col: chromadb.Collection, batch: list[Chunk]) -> None:
        col.add(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            embeddings=self.embedder.embed_documents([c.text for c in batch]),
            metadatas=[c.metadata() for c in batch],
        )

    def drop(self, name: str) -> None:
        try:
            self._client.delete_collection(name)
        except NotFoundError:
            pass

    def exists(self, name: str) -> bool:
        try:
            self._client.get_collection(name)
            return True
        except NotFoundError:
            return False

    def model_of(self, name: str) -> str:
        """Embedding model a collection was written with; none recorded = minilm (D59)."""
        meta = self._client.get_collection(name).metadata or {}
        return str(meta.get("embedding_model", DEFAULT_MODEL))

    def check_model(self, name: str) -> None:
        stored = self.model_of(name)
        if stored != self.embedder.name:
            raise EmbeddingModelMismatch(name, stored, self.embedder.name)

    def query(self, name: str, text: str, k: int) -> list[RetrievedChunk]:
        col = self._client.get_collection(name)
        stored = str((col.metadata or {}).get("embedding_model", DEFAULT_MODEL))
        if stored != self.embedder.name:  # never compare vectors from two different models
            raise EmbeddingModelMismatch(name, stored, self.embedder.name)
        res = col.query(
            query_embeddings=[self.embedder.embed_query(text)],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            _to_chunk(doc, meta, 1.0 - float(dist))  # cosine similarity
            for doc, meta, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True
            )
        ]

    def all_chunks(self, name: str) -> list[RetrievedChunk]:
        """Every chunk of a collection, exactly as embedded (for the BM25 index, D55)."""
        res = self._client.get_collection(name).get(include=["documents", "metadatas"])
        return [
            _to_chunk(doc, meta, 0.0)
            for doc, meta in zip(res["documents"], res["metadatas"], strict=True)
        ]


def _to_chunk(doc: str, meta, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=str(meta["chunk_id"]),
        company_id=str(meta["company_id"]),
        company_name=str(meta["company_name"]),
        version=int(meta["version"]),
        doc_title=str(meta["doc_title"]),
        section_id=str(meta["section_id"]),
        section_title=str(meta["section_title"]),
        text=doc,
        score=score,
    )
