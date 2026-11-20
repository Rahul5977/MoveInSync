"""Eval runner on SYNTHETIC cases (not eval/cases.jsonl): row format, D38e routing, phases,
report metrics and the tau sweep. Keyless (extractive fake)."""

import json

import pytest

from app.llm.fake import ExtractiveLLM
from eval import run_eval
from tests.conftest import POLICIES, make_test_settings

ESCORT = "Do I get a security escort for a 7:45 PM drop? I'm a woman employee."


def _case(id, company, q, status, phase="seed", sections=(), inc=(), exc=(), cat="direct", **kw):
    exp = {
        "status": status,
        "sections": list(sections),
        "require_all_sections": False,
        "version": 2 if phase == "after_nimbus_v2" else 1,
        "must_include": [list(g) for g in inc],
        "must_not_include": list(exc),
        **kw,
    }
    return {
        "id": id,
        "input": {"company": company, "question": q, "phase": phase},
        "expected": exp,
        "notes": f"category: {cat}",
    }


CASES = [
    _case("v1", "nimbus", ESCORT, "answered", sections=["8"], inc=[["8:00 PM"]], exc=["7:30"]),
    _case("v2", "nimbus", ESCORT, "answered", "after_nimbus_v2", ["8"], [["7:30 PM"]], ["8:00 PM"]),
    _case(
        "junk",
        "kestrel",
        "What is the capital of France?",
        "not_covered",
        cat="universal-silence",
        gate="tau",
    ),
]


@pytest.fixture
def settings(tmp_path):
    return make_test_settings(data_dir=tmp_path / "unused", policies_dir=POLICIES)


def test_output_paths_fake_never_writes_submission_file():
    assert run_eval.output_paths("fake")[0].name == "results_fake.jsonl"
    assert run_eval.output_paths("groq")[0].name == "results.jsonl"


def test_choose_provider_auto(settings):
    assert run_eval.choose_provider("auto", settings) == "fake"  # no key
    with_key = settings.model_copy(update={"groq_api_key": "gsk_x"})
    assert run_eval.choose_provider("auto", with_key) == "groq"
    assert run_eval.choose_provider("fake", with_key) == "fake"


def test_run_once_phases_rows_and_report(settings):
    results = run_eval.run_once(settings, CASES, ExtractiveLLM(), throttle_s=0)
    by_id = {c["id"]: (c, a, s) for c, a, s, _ in results}

    assert by_id["v1"][1]["citations"] == ["nimbus_v1_s8_p0"] and by_id["v1"][2].passed
    assert by_id["v2"][1]["citations"] == ["nimbus_v2_s8_p0"] and by_id["v2"][2].passed
    assert by_id["junk"][1]["gate"] == "tau" and by_id["junk"][1]["llm_calls"] == 0

    row = run_eval.to_row(*results[0], provider="fake", model="m")
    assert set(row) == {"id", "input", "expected", "actual", "pass", "notes"}
    assert "provider=fake" in row["notes"] and json.dumps(row)  # serialisable

    rep = run_eval.build_report(results, {"provider": "fake"})
    assert rep["summary"] == {"total": 3, "passed": 3, "pass_rate": 1.0}
    assert rep["metrics"]["leaks"] == 0 and rep["metrics"]["max_llm_calls"] <= 2
    assert rep["metrics"]["retrieval_hit_at_3"] == 1.0
    assert rep["metrics"]["not_covered_by_gate"] == {"tau": 1}


def test_sweep_picks_highest_tau_without_false_refusals(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "EVAL_DIR", tmp_path)
    out = run_eval.run_sweep(settings, CASES)
    answerable = [out["best_scores"]["v1"], out["best_scores"]["v2"]]
    assert out["chosen_tau"] <= min(answerable)
    row = next(r for r in out["sweep"] if r["tau"] == out["chosen_tau"])
    assert row["false_refusals"] == [] and "junk" in row["not_covered_caught"]
    assert (tmp_path / "tau_sweep.json").exists()


def test_real_cases_file_is_valid():
    """eval/cases.jsonl: unique ids, known phases, D51 (no question terms forbidden)."""
    cases = run_eval.load_cases()
    assert len(cases) == 33
    assert run_eval.question_terms_forbidden(cases) == {}


def test_d51_violation_detected():
    bad = _case("x", "nimbus", "Is it after 10 PM?", "answered", exc=["10 PM"])
    assert run_eval.question_terms_forbidden([bad]) == {"x": ["10 PM"]}
