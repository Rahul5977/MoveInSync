"""Unit tests for the eval scoring rules (D48, D49). Pure functions, no I/O."""

import pytest

from eval.scoring import contains, normalise, parse_chunk_id, percentile, score_case


@pytest.mark.parametrize(
    ("text", "phrase", "hit"),
    [
        ("Capped at ₹1,500 per trip.", "1500", True),
        ("Capped at Rs. 1,500 per trip.", "₹1500", True),
        ("Capped at INR 1500.", "rs 1500", True),
        ("An escort after 8:00 P.M. is required", "8:00 pm", True),
        ("Approved by Transport Admin and HR.", "hr", True),
        ("Three no-shows remove you.", "hr", False),  # word boundary: not inside "three"
        ("Wait up to 30 minutes.", "3", False),  # not inside "30"
        ("The 3rd no-show", "3", False),
        ("No, Kestrel does not provide this.", "no", True),
        ("Nothing is said.", "no", False),  # not inside "nothing"
        ("Escort after 10:00 PM.", "10 PM", False),  # different phrase; alternatives list both
    ],
)
def test_contains_normalised_word_boundaries(text, phrase, hit):
    assert contains(text, phrase) is hit


def test_normalise():
    assert normalise("  Rs.2,000  P.M. ") == "₹2000 pm"


def test_parse_chunk_id():
    assert parse_chunk_id("nimbus_v2_s8_p0") == ("nimbus", 2, "8")
    assert parse_chunk_id("saffron_v1_s4.2_p1") == ("saffron", 1, "4.2")
    assert parse_chunk_id("garbage") is None


def _case(**exp):
    base = {
        "status": "answered",
        "sections": ["8"],
        "require_all_sections": False,
        "version": 1,
        "must_include": [["8:00 PM", "8 PM"]],
        "must_not_include": ["10 PM"],
    }
    return {"id": "x", "input": {"company": "nimbus"}, "expected": {**base, **exp}}


def _actual(**kw):
    base = {
        "status": "answered",
        "answer": "Yes, an escort is required after 8:00 PM.",
        "citations": ["nimbus_v1_s8_p0"],
        "retrieved_ids": ["nimbus_v1_s8_p0", "nimbus_v1_s7_p0", "nimbus_v1_s9_p0"],
        "gate": "answered",
    }
    return {**base, **kw}


def test_pass():
    s = score_case(_case(), _actual())
    assert s.passed and s.sections_ok and s.facts_ok and s.hit_at_3 and not s.leak


def test_wrong_version_citation_fails():
    s = score_case(_case(version=2), _actual())
    assert not s.passed and s.sections_ok is False


def test_forbidden_phrase_fails():
    s = score_case(_case(), _actual(answer="After 8 PM (not 10 PM)."))
    assert not s.passed and "contains forbidden '10 PM'" in s.failures


def test_require_all_sections():
    case = _case(
        sections=["10", "11"], require_all_sections=True, must_include=[], must_not_include=[]
    )
    one = _actual(citations=["nimbus_v1_s10_p0"], retrieved_ids=["nimbus_v1_s10_p0"])
    both = _actual(citations=["nimbus_v1_s10_p0", "nimbus_v1_s11_p0"], retrieved_ids=[])
    assert not score_case(case, one).passed
    assert score_case(case, both).passed


def test_leak_fails_even_if_answer_right():
    s = score_case(_case(), _actual(retrieved_ids=["kestrel_v1_s8_p0"]))
    assert s.leak and not s.passed


def test_not_covered_case_and_gate_reported_not_pass_critical():
    case = _case(
        status="not_covered", sections=[], must_include=[], must_not_include=[], gate="tau"
    )
    actual = _actual(
        status="not_covered", answer="doesn't cover this", citations=[], gate="llm_not_covered"
    )
    s = score_case(case, actual)
    assert s.passed and s.gate_as_expected is False and s.sections_ok is None


def test_status_mismatch_fails():
    s = score_case(_case(status="not_covered", sections=[], must_include=[]), _actual())
    assert not s.passed and s.failures[0].startswith("status answered")


def test_percentile_nearest_rank():
    vals = list(range(1, 101))
    assert percentile(vals, 50) == 50 and percentile(vals, 95) == 95 and percentile([], 95) is None
    assert percentile([300, 100, 200], 95) == 300


def test_budget_conditions_are_hard_on_every_case():
    three = ["nimbus_v1_s8_p0", "nimbus_v1_s7_p0", "nimbus_v1_s9_p0"]
    assert score_case(_case(), _actual(llm_calls=2, citations=three)).passed
    over_calls = score_case(_case(), _actual(llm_calls=3))
    assert not over_calls.passed and over_calls.failures[0].startswith("BUDGET")
    over_cites = score_case(_case(), _actual(citations=[*three, "nimbus_v1_s10_p0"]))
    assert not over_cites.passed
    nc = _case(status="not_covered", sections=[], must_include=[], must_not_include=[])
    assert not score_case(nc, _actual(status="not_covered", citations=[], llm_calls=3)).passed
