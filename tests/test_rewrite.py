"""Query rewriting on a miss (D61), scripted with FakeLLM. No real LLM, no network."""

import json

import pytest

from app.llm.base import ConnectionDropped, LLMError
from app.llm.fake import FakeLLM, sections_in
from app.rag.answerer import NOT_COVERED_MSG, answer_question
from app.schemas import Principal
from tests.test_answerer import ESCORT_Q, FakeClock, app_with

SQ = "airport trip approval and advance booking"  # pulls different sections than ESCORT_Q
MARKER = "zqx-marker-term"  # appears only in search_query, to prove it never reaches a prompt


def miss(search_query=SQ) -> str:
    body = {"covered": False, "answer": "", "cited_ids": []}
    if search_query is not ...:
        body["search_query"] = search_query
    return json.dumps(body)


def hit(user: str) -> str:
    cid = sections_in(user)[0][0]
    return json.dumps({"covered": True, "answer": "Per policy.", "cited_ids": [cid]})


def run(settings, llm, question=ESCORT_Q, company="nimbus", clock=None, rewrite="on"):
    s = settings.model_copy(update={"rewrite_on_miss": rewrite})
    with app_with(s, FakeLLM([])) as client:
        st = client.app.state
        return answer_question(
            Principal(sub="t", company_id=company, role="employee"),
            question,
            st.registry,
            st.store,
            llm,
            s,
            clock=clock or FakeClock(),
            bm25=st.bm25,
        )


def scripted(*steps):
    """FakeLLM whose replies are functions of the user prompt (or exceptions)."""
    steps = list(steps)

    def _r(system, user):
        nxt = steps.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt(user) if callable(nxt) else nxt

    return FakeLLM(_r)


# ---------- the happy path ----------


def test_round1_miss_then_round2_answers_with_two_calls(settings):
    llm = scripted(miss(), hit)
    resp, trace = run(settings, llm)
    assert resp.status == "answered" and resp.meta.llm_calls == 2
    assert trace.rewrite_triggered and trace.search_query == SQ
    assert trace.round2_sent_ids != trace.retrieved_ids[:3]
    assert {c.chunk_id for c in resp.citations} <= set(trace.round2_sent_ids)
    # call #2 answers the ORIGINAL question with the new sections
    assert ESCORT_Q in llm.calls[1][1]
    assert [cid for cid, _ in sections_in(llm.calls[1][1])] == trace.round2_sent_ids


# ---------- round 1 used its second call: no rewrite ----------


def test_invalid_json_round1_repairs_and_never_rewrites(settings):
    llm = scripted("not json", miss())
    resp, trace = run(settings, llm)
    assert resp.status == "not_covered" and resp.meta.llm_calls == 2
    assert not trace.rewrite_triggered and trace.rewrite_skip_reason == "round1_used_2_calls"


def test_connection_drop_round1_retries_and_never_rewrites(settings):
    llm = scripted(ConnectionDropped("network: ConnectError"), miss())
    resp, trace = run(settings, llm)
    assert resp.status == "not_covered" and resp.meta.llm_calls == 2
    assert trace.retried and trace.rewrite_skip_reason == "round1_used_2_calls"


# ---------- round 2 failures are not_covered ----------


@pytest.mark.parametrize(
    "second",
    [LLMError("timeout"), ConnectionDropped("network: ReadError"), "junk", miss()],
    ids=["timeout", "connection", "bad_json", "covered_false"],
)
def test_round2_failure_is_not_covered(settings, second):
    resp, trace = run(settings, scripted(miss(), second))
    assert resp.status == "not_covered" and resp.answer == NOT_COVERED_MSG
    assert resp.meta.llm_calls == 2 and resp.citations == []
    assert trace.gate in ("rewrite_failed", "rewrite_not_covered")


# ---------- gates that prevent call #2 ----------


def test_identical_round2_top3_makes_no_second_call(settings):
    llm = scripted(miss(search_query=ESCORT_Q))  # same words -> same top 3
    resp, trace = run(settings, llm)
    assert resp.status == "not_covered" and resp.meta.llm_calls == 1 and len(llm.calls) == 1
    assert trace.rewrite_skip_reason == "same_top3"


def test_less_than_1_2_s_left_makes_no_second_call(settings):
    clock = FakeClock()

    def slow_miss(user):
        clock.t += 1.7  # 2.8 - 1.7 = 1.1 s left
        return miss()

    llm = scripted(slow_miss)
    resp, trace = run(settings, llm, clock=clock)
    assert resp.status == "not_covered" and resp.meta.llm_calls == 1
    assert trace.rewrite_skip_reason == "deadline"


def test_tau_refusal_still_makes_zero_calls(settings):
    llm = FakeLLM([])
    resp, trace = run(settings, llm, question="Can I bring my pet dog in the cab?")
    assert trace.gate == "tau" and resp.meta.llm_calls == 0 and llm.calls == []


@pytest.mark.parametrize(
    "bad", [..., "", "   ", "x" * 201, 42, ["airport"]], ids=lambda v: repr(v)[:12]
)
def test_invalid_search_query_counts_as_absent(settings, bad):
    llm = scripted(miss(search_query=bad))
    resp, trace = run(settings, llm)
    assert resp.status == "not_covered" and resp.meta.llm_calls == 1
    assert trace.search_query is None and trace.rewrite_skip_reason == "no_search_query"


# ---------- isolation and secrecy of the rewrite ----------


def test_search_query_naming_another_company_stays_in_callers_collection(settings):
    llm = scripted(miss(search_query="Saffron Pharma ad-hoc trip cost cap approval"), hit)
    resp, trace = run(settings, llm, company="nimbus")
    assert trace.round2_retrieved_ids
    assert all(cid.startswith("nimbus_v1_") for cid in trace.round2_retrieved_ids)
    for _, user in llm.calls:
        assert all(cid.startswith("nimbus_v1_") for cid, _ in sections_in(user))


def test_search_query_never_in_a_prompt_or_the_response(settings):
    llm = scripted(miss(search_query=f"airport trip approval {MARKER}"), hit)
    resp, trace = run(settings, llm)
    assert trace.rewrite_triggered
    for system, user in llm.calls:
        assert MARKER not in system and MARKER not in user
    assert MARKER not in resp.model_dump_json()


def test_rewrite_off_never_adds_the_rule_or_a_second_call(settings):
    llm = scripted(miss())
    resp, trace = run(settings, llm, rewrite="off")
    assert resp.status == "not_covered" and resp.meta.llm_calls == 1
    assert "search_query" not in llm.calls[0][0] and trace.search_query is None


# ---------- property: the budget holds in every scripted scenario ----------

SCENARIOS = {
    "answer_round1": [hit],
    "miss_then_answer": [miss(), hit],
    "miss_then_timeout": [miss(), LLMError("timeout")],
    "miss_then_junk": [miss(), "junk"],
    "miss_then_miss": [miss(), miss()],
    "junk_then_miss": ["junk", miss()],
    "junk_then_junk": ["junk", "junk"],
    "drop_then_miss": [ConnectionDropped("network: ConnectError"), miss()],
    "drop_then_answer": [ConnectionDropped("network: ConnectError"), hit],
    "miss_without_query": [miss(search_query=...)],
    "miss_same_top3": [miss(search_query=ESCORT_Q)],
    "miss_then_answer_citing_everything": [
        miss(),
        lambda u: json.dumps(
            {"covered": True, "answer": "x", "cited_ids": [c for c, _ in sections_in(u)] * 3}
        ),
    ],
    "miss_then_answer_citing_unknown": [
        miss(),
        json.dumps({"covered": True, "answer": "x", "cited_ids": ["kestrel_v1_s3_p0"]}),
    ],
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_budget_property_calls_and_citations(settings, name):
    llm = scripted(*SCENARIOS[name])
    resp, _ = run(settings, llm)
    assert resp.meta.llm_calls <= 2 and len(llm.calls) <= 2
    assert len(resp.citations) <= 3
    assert resp.meta.llm_calls == len(llm.calls)
