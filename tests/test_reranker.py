"""Cross-encoder reranking (D60). No network: model files are baked into the Docker test image
(or fetched once locally with `python -m scripts.fetch_models`)."""

import pytest

from app.llm.fake import FakeLLM, sections_in
from app.rag.answerer import answer_question
from app.rag.reranker import CE_FILES, CrossEncoder, Reranker
from app.schemas import Principal, RetrievedChunk
from tests.conftest import REPO, read_policy
from tests.test_answerer import ESCORT_Q, app_with, ask, reply

CE_DIR = REPO / "models" / "ms-marco-MiniLM-L6-v2"
pytestmark = pytest.mark.skipif(
    not all((CE_DIR / f).is_file() for f in CE_FILES),
    reason="cross-encoder files not present (run: uv run python -m scripts.fetch_models)",
)


@pytest.fixture(scope="module")
def model():
    return CrossEncoder(CE_DIR)


def _chunk(i: int, company: str = "nimbus", version: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{company}_v{version}_s{i}_p0",
        company_id=company,
        company_name="X",
        version=version,
        doc_title="T",
        section_id=str(i),
        section_title="S",
        text=f"section {i}",
        score=0.0,
    )


def test_scores_are_deterministic(model):
    texts = ["Cabs wait 5 minutes at pickup.", "The cafeteria opens at noon."]
    assert model.score("How long does the cab wait?", texts) == model.score(
        "How long does the cab wait?", texts
    )


def test_relevant_passage_outranks_unrelated(model):
    rel, unrel = model.score(
        "Will I be charged if I miss my cab?",
        [
            "A no-show fee is deducted when the employee does not board the booked cab.",
            "The cafeteria serves lunch between noon and 2 PM.",
        ],
    )
    assert rel > unrel


def test_candidates_capped_at_40(model):
    class Store:
        def all_chunks(self, name):
            return [_chunk(i) for i in range(50, 0, -1)]

    cands = Reranker(model, cap=40).build(Store(), "nimbus_v1")
    assert len(cands) == 40
    assert [c.chunk_id for c in cands] == sorted(c.chunk_id for c in cands)


def _rerank_settings(settings):
    return settings.model_copy(update={"rerank": "on", "reranker_model_dir": CE_DIR})


def test_only_active_collection_chunks_are_reranked(settings, token_for):
    llm = FakeLLM(reply())
    with app_with(_rerank_settings(settings), llm) as client:
        r = ask(client, "kestrel", ESCORT_Q, token_for)
        st = client.app.state
        assert st.reranker.names() == ["kestrel_v1", "nimbus_v1", "saffron_v1"]
    assert r.status_code == 200
    sent = sections_in(llm.calls[0][1])
    assert len(sent) == 3 and all(cid.startswith("kestrel_v1_") for cid, _ in sent)


def test_after_v2_activation_only_v2_chunks_are_candidates(settings, token_for):
    llm = FakeLLM(reply())
    with app_with(_rerank_settings(settings), llm) as client:
        up = client.post(
            "/admin/ingest",
            files={"file": ("v2.md", read_policy("updates/nimbus/v2.md"), "text/markdown")},
            headers=token_for("nimbus", "admin"),
        )
        assert up.status_code == 201
        assert "nimbus_v1" not in client.app.state.reranker.names()
        r = ask(client, "nimbus", ESCORT_Q, token_for)
    assert r.json()["meta"]["policy_version"] == 2
    assert all(cid.startswith("nimbus_v2_") for cid, _ in sections_in(llm.calls[0][1]))


def _answer(settings, question, reranker, llm):
    p = Principal(sub="t", company_id="nimbus", role="employee")
    with app_with(settings, FakeLLM([])) as client:  # startup seeds nothing new; reuse stores
        st = client.app.state
        return answer_question(
            p, question, st.registry, st.store, llm, settings, bm25=st.bm25, reranker=reranker
        )


def test_rerank_adds_no_llm_call_and_records_scores(settings, model):
    s = _rerank_settings(settings)
    llm = FakeLLM(reply())
    resp, trace = _answer(s, ESCORT_Q, Reranker(model), llm)
    assert resp.meta.llm_calls == 1 and len(llm.calls) == 1
    assert trace.rerank_scores and set(trace.rerank_scores) == set(trace.retrieved_ids)
    assert trace.sent_ids[0] == "nimbus_v1_s8_p0"  # escort section first


def test_tau_decision_unchanged_by_rerank(settings, model):
    q = "Can I bring my pet dog in the cab?"
    off_resp, off_trace = _answer(settings, q, None, FakeLLM([]))
    on_resp, on_trace = _answer(_rerank_settings(settings), q, Reranker(model), FakeLLM([]))
    assert off_trace.gate == on_trace.gate == "tau"
    assert off_trace.best_score == on_trace.best_score
    assert off_resp.meta.llm_calls == on_resp.meta.llm_calls == 0


def test_rerank_off_ignores_a_given_reranker(settings, model):
    """RERANK=off: even with a reranker object around, results equal plain vector retrieval."""
    a_resp, a_trace = _answer(settings, ESCORT_Q, None, FakeLLM(reply()))
    b_resp, b_trace = _answer(settings, ESCORT_Q, Reranker(model), FakeLLM(reply()))
    assert a_trace.retrieved_ids == b_trace.retrieved_ids and b_trace.rerank_scores is None
    assert a_resp.citations == b_resp.citations and a_resp.status == b_resp.status
