"""Hybrid retrieval (D55-D57): BM25 setup, RRF, isolation, versioning, mode switch. Keyless."""

from contextlib import contextmanager

from fastapi.testclient import TestClient

from app.llm.fake import ExtractiveLLM
from app.main import create_app
from app.rag.bm25 import Bm25Cache, Bm25Index, tokenize
from app.rag.retriever import rrf_merge, search
from app.schemas import RetrievedChunk
from tests.conftest import read_policy

COMPANIES = ("nimbus", "kestrel", "saffron")
# Names other companies and uses their vocabulary, to tempt both retrievers across tenants.
CROSS_Q = "Nimbus Software Kestrel Bank Saffron Pharma escort 8:00 PM ad-hoc cap ₹2,000 Saturday"


@contextmanager
def hybrid_app(settings, mode="hybrid"):
    s = settings.model_copy(update={"retrieval_mode": mode})
    with TestClient(create_app(s, llm=ExtractiveLLM())) as client:
        yield client


def _chunk(cid, text, score=0.0, company="nimbus", version=1):
    return RetrievedChunk(
        chunk_id=cid,
        company_id=company,
        company_name="N",
        version=version,
        doc_title="T",
        section_id="1",
        section_title="S",
        text=text,
        score=score,
    )


# ---------- D55: fixed setup ----------


def test_tokenize_lowercase_words_stopwords_then_snowball():
    assert tokenize("The No-show Policy: will I be FINED for Requests?") == [
        "show",
        "polici",
        "fine",
        "request",
    ]


def test_bm25_returns_only_matching_chunks_best_first():
    idx = Bm25Index(
        [
            _chunk("a", "Saturday transport needs manager approval"),
            _chunk("b", "Escort after 8 PM"),
            _chunk("c", "Saturday Saturday rules"),
        ]
    )
    hits = idx.search("saturday", 8)
    assert [h.chunk_id for h in hits] == ["c", "a"] and all(h.score > 0 for h in hits)
    assert idx.search("the of and", 8) == []  # stopwords only -> nothing


def test_rrf_merge_sums_reciprocal_ranks_and_breaks_ties_by_vector_rank():
    v = [_chunk("x", ""), _chunk("y", "")]
    lex = [_chunk("y", ""), _chunk("z", "")]
    merged = rrf_merge(v, lex, k=60)
    assert [c.chunk_id for c in merged] == ["y", "x", "z"]  # y in both lists wins
    assert abs(merged[0].score - (1 / 62 + 1 / 61)) < 1e-12
    tie = rrf_merge([_chunk("p", "")], [_chunk("q", "")], k=60)
    assert [c.chunk_id for c in tie] == ["p", "q"]  # equal score -> vector rank first


# ---------- isolation and versioning ----------


def test_hybrid_isolation_no_foreign_chunks(settings, token_for):
    with hybrid_app(settings) as client:
        st = client.app.state
        for co in COMPANIES:
            r = search(co, CROSS_Q, st.registry, st.store, 8, mode="hybrid", bm25=st.bm25)
            assert r.chunks and {c.company_id for c in r.chunks} == {co}
            lexical = st.bm25.get(st.store, st.registry.active(co).collection_name)
            assert {c.company_id for c in lexical.search(CROSS_Q, 8)} == {co}
            body = client.post("/ask", json={"question": CROSS_Q}, headers=token_for(co)).json()
            assert all(c["chunk_id"].startswith(f"{co}_") for c in body["citations"])


def test_hybrid_versioning_only_v2_after_ingest_including_bm25(settings, token_for):
    q = "How many hours before the trip must I submit a roster change?"
    with hybrid_app(settings) as client:
        st = client.app.state
        assert "nimbus_v1" in st.bm25.names()
        r = client.post(
            "/admin/ingest",
            files={"file": ("v2.md", read_policy("updates/nimbus/v2.md"), "text/markdown")},
            headers=token_for("nimbus", "admin"),
        )
        assert r.status_code == 201
        assert "nimbus_v2" in st.bm25.names() and "nimbus_v1" not in st.bm25.names()
        lexical = st.bm25.get(st.store, "nimbus_v2").search(q, 8)
        assert lexical and {c.version for c in lexical} == {2}
        res = search("nimbus", q, st.registry, st.store, 8, mode="hybrid", bm25=st.bm25)
        assert res.version == 2 and {c.version for c in res.chunks} == {2}
        body = client.post("/ask", json={"question": q}, headers=token_for("nimbus")).json()
    assert body["citations"][0]["chunk_id"].startswith("nimbus_v2_")
    assert "3 hours" in body["answer"]


# ---------- RETRIEVAL_MODE switch ----------


class _ExplodingCache(Bm25Cache):
    def get(self, store, name):
        raise AssertionError("vector mode must not touch BM25")


def test_retrieval_mode_switch(settings, token_for):
    q = "can i get cab on sat?"
    for mode in ("vector", "hybrid"):
        with hybrid_app(settings, mode) as client:
            body = client.post("/ask", json={"question": q}, headers=token_for("nimbus")).json()
            assert body["meta"]["retrieval_mode"] == mode
    with hybrid_app(settings, "vector") as client:
        st = client.app.state
        vec = search("nimbus", q, st.registry, st.store, 8, bm25=_ExplodingCache())
        hyb = search("nimbus", q, st.registry, st.store, 8, mode="hybrid", bm25=st.bm25)
    # tau input is the best VECTOR score in both modes (D56); only the order may differ.
    assert vec.best_vector_score == hyb.best_vector_score == vec.chunks[0].score
