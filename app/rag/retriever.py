"""Retrieval restricted to the caller's company's ACTIVE collection.

The collection comes from the registry pointer for `company_id` (taken from the JWT). There is
no parameter through which a client can name a collection, company or version.
Returns the whole candidate pool (D22), best first; the answerer decides what to send.

Hybrid mode (D56): vector and BM25 each fetch their own top `k` from the same active collection,
merged with Reciprocal Rank Fusion. tau stays on the best *vector* score.

Rerank (D60): candidates = ALL chunks of the active collection (cached, cap 40), sorted by a
cross-encoder; the vector query still runs, only for tau (best MiniLM score of the question).
"""

import logging
from typing import NamedTuple

from app.rag.bm25 import Bm25Cache
from app.rag.reranker import Reranker
from app.schemas import RetrievedChunk
from app.store.registry import Registry
from app.store.vector import VectorStore

log = logging.getLogger(__name__)


def retrieve(
    company_id: str, question: str, registry: Registry, store: VectorStore, k: int
) -> tuple[list[RetrievedChunk], int | None]:
    """Vector-only shorthand: (chunks best-first, active_version); ([], None) if no live policy."""
    chunks, version, _ = search(company_id, question, registry, store, k)
    return chunks, version


class Retrieval(NamedTuple):
    chunks: list[RetrievedChunk]  # fused order in hybrid mode, vector order otherwise
    version: int | None
    best_vector_score: float | None  # what tau is applied to, in both modes


def rrf_merge(
    vector: list[RetrievedChunk], lexical: list[RetrievedChunk], k: int
) -> list[RetrievedChunk]:
    """score = sum of 1/(k + rank) over the lists a chunk appears in (ranks from 1).
    Ties: better vector rank first, then chunk id. Returned chunks carry the RRF score."""
    by_id: dict[str, RetrievedChunk] = {}
    fused: dict[str, float] = {}
    vrank: dict[str, int] = {}
    for ranked, is_vector in ((vector, True), (lexical, False)):
        for rank, c in enumerate(ranked, 1):
            by_id.setdefault(c.chunk_id, c)
            fused[c.chunk_id] = fused.get(c.chunk_id, 0.0) + 1.0 / (k + rank)
            if is_vector:
                vrank[c.chunk_id] = rank
    order = sorted(fused, key=lambda cid: (-fused[cid], vrank.get(cid, 10**6), cid))
    return [by_id[cid].model_copy(update={"score": fused[cid]}) for cid in order]


def search(
    company_id: str,
    question: str,
    registry: Registry,
    store: VectorStore,
    k: int,
    mode: str = "vector",
    bm25: Bm25Cache | None = None,
    rrf_k: int = 60,
    reranker: Reranker | None = None,
) -> Retrieval:
    active = registry.active(company_id)
    if active is None:
        return Retrieval([], None, None)
    vector = sorted(
        store.query(active.collection_name, question, k), key=lambda c: c.score, reverse=True
    )
    best = vector[0].score if vector else None
    if reranker is not None:  # D60: rerank every chunk of the active collection
        chunks = reranker.rerank(question, reranker.candidates(store, active.collection_name))
    elif mode == "hybrid":
        index = (bm25 or Bm25Cache()).get(store, active.collection_name)
        chunks = rrf_merge(vector, index.search(question, k), rrf_k)
    else:
        chunks = vector
    # Defence in depth on the (merged) list: own company, active version only.
    safe = [c for c in chunks if c.company_id == company_id and c.version == active.version]
    if len(safe) != len(chunks):
        log.error("Isolation filter dropped %d foreign chunks", len(chunks) - len(safe))
    return Retrieval(safe, active.version, best)
