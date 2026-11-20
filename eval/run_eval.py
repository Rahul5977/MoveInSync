"""Run the fixed evaluation set, or the retrieval-only tau sweep.

Usage:
  uv run python -m eval.run_eval                  # provider: groq if GROQ_API_KEY is set, else fake
  uv run python -m eval.run_eval --llm fake       # keyless; writes eval/results_fake.jsonl
  uv run python -m eval.run_eval --llm groq       # writes eval/results.jsonl (the submission file)
  uv run python -m eval.run_eval --sweep          # tau sweep, retrieval only, no LLM
  uv run python -m eval.run_eval --repeat 2       # run twice, report cases whose pass/fail flips

Every run starts from a FRESH temporary data dir: seed v1 of every company, run the `seed` cases,
ingest Nimbus v2 through the real ingestion path, then run the `after_nimbus_v2` cases. So results
are reproducible and never depend on local state.

Output rows have exactly: id, input, expected, actual, pass, notes (fixed by the brief).
D38e: a fake run can never write eval/results.jsonl; the provider is in every row's notes and in
the report, and a banner says which provider ran.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from app.config import Settings, get_settings
from app.embeddings import make_embedder
from app.ingest.service import ingest_policy, seed_policies
from app.llm.base import LLMClient
from app.llm.factory import make_llm
from app.rag.answerer import answer_question
from app.rag.bm25 import Bm25Cache
from app.rag.retriever import retrieve, search
from app.schemas import Principal
from app.store.registry import Registry
from app.store.vector import VectorStore
from eval.scoring import contains, parse_chunk_id, percentile, score_case

EVAL_DIR = Path(__file__).resolve().parent
CASES = EVAL_DIR / "cases.jsonl"
PHASES = ("seed", "after_nimbus_v2")
V2_UPDATE = ("nimbus", "updates/nimbus/v2.md")
RATE_LIMIT_RETRIES, RATE_LIMIT_WAIT_S = 3, 15.0


# ---------------------------------------------------------------- setup


def load_cases(path: Path = CASES) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        sys.exit("Duplicate case ids in cases.jsonl")
    bad = [r["id"] for r in rows if r["input"]["phase"] not in PHASES]
    if bad:
        sys.exit(f"Unknown phase in cases: {bad}")
    violations = question_terms_forbidden(rows)
    if violations:
        sys.exit(f"D51: must_not_include terms that appear in the question: {violations}")
    return rows


def question_terms_forbidden(rows: list[dict]) -> dict[str, list[str]]:
    """D51: a correct answer may echo the question, so its terms can't be forbidden."""
    out = {}
    for r in rows:
        hits = [t for t in r["expected"]["must_not_include"] if contains(r["input"]["question"], t)]
        if hits:
            out[r["id"]] = hits
    return out


def choose_provider(requested: str, settings: Settings) -> str:
    if requested != "auto":
        return requested
    return "groq" if settings.groq_api_key else "fake"


def variant_suffix(mode: str = "vector", embedding: str = "minilm") -> str:
    """'' for the submission setup (vector + minilm); e.g. '_hybrid', '_bge', '_hybrid_bge'."""
    parts = ([mode] if mode != "vector" else []) + (["bge"] if embedding == "bge-small" else [])
    return "".join(f"_{p}" for p in parts)


def output_paths(
    provider: str, mode: str = "vector", embedding: str = "minilm"
) -> tuple[Path, Path]:
    """Only groq + vector + minilm writes the submission file; other setups get a suffix."""
    suffix = variant_suffix(mode, embedding)
    if provider == "groq":
        return EVAL_DIR / f"results{suffix}.jsonl", EVAL_DIR / f"report{suffix}.json"
    return EVAL_DIR / f"results_fake{suffix}.jsonl", EVAL_DIR / f"report_fake{suffix}.json"


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class Env:
    """A fresh, private copy of the system: registry + vector store in a temp dir."""

    def __init__(self, settings: Settings) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="eval_var_")
        self.settings = settings.model_copy(update={"data_dir": Path(self._tmp.name)})
        self.registry = Registry(self.settings.registry_path)
        self.registry.init_schema()
        self.store = VectorStore(
            self.settings.chroma_dir,
            make_embedder(self.settings.embedding_model, self.settings.bge_model_dir),
        )
        self.bm25 = Bm25Cache()  # used only in hybrid mode; builds per collection on first use
        outcomes = seed_policies(self.registry, self.store, self.settings)
        if not all(o.startswith("ingested") for o in outcomes.values()):
            sys.exit(f"Seeding failed: {outcomes}")

    def ingest_update(self) -> None:
        company, rel = V2_UPDATE
        text = (self.settings.policies_dir / rel).read_text(encoding="utf-8")
        ingest_policy(company, text, self.registry, self.store, self.settings)

    def close(self) -> None:
        self._tmp.cleanup()


def banner(lines: list[str]) -> None:
    width = max(len(x) for x in lines) + 4
    print("=" * width)
    for x in lines:
        print(f"  {x}")
    print("=" * width)


# ---------------------------------------------------------------- tau sweep (no LLM)


def run_sweep(settings: Settings, cases: list[dict]) -> dict:
    env = Env(settings)
    scores: dict[str, float] = {}
    try:
        for phase in PHASES:
            if phase == "after_nimbus_v2":
                env.ingest_update()
            for c in (c for c in cases if c["input"]["phase"] == phase):
                chunks, _ = retrieve(
                    c["input"]["company"],
                    c["input"]["question"],
                    env.registry,
                    env.store,
                    settings.retrieval_pool,
                )
                scores[c["id"]] = chunks[0].score if chunks else 0.0
    finally:
        env.close()

    answerable = [c["id"] for c in cases if c["expected"]["status"] == "answered"]
    not_cov = [c["id"] for c in cases if c["expected"]["status"] == "not_covered"]
    universal = [c["id"] for c in cases if "universal-silence" in c.get("notes", "")]

    grid = [round(t * 0.01, 2) for t in range(0, 61)]
    rows = []
    for tau in grid:
        rows.append(
            {
                "tau": tau,
                "false_refusals": sorted(i for i in answerable if scores[i] < tau),
                "not_covered_caught": sorted(i for i in not_cov if scores[i] < tau),
                "universal_caught": sorted(i for i in universal if scores[i] < tau),
            }
        )
    safe = [r["tau"] for r in rows if not r["false_refusals"]]
    chosen = max(safe) if safe else 0.0
    lowest_answerable = min((scores[i], i) for i in answerable)
    highest_off_topic = max((scores[i], i) for i in universal) if universal else None

    print(
        f"\n{'tau':>5} | false refusals | not-covered caught (of {len(not_cov)})"
        f" | universal caught (of {len(universal)})"
    )
    prev = None
    for r in rows:
        key = (len(r["false_refusals"]), len(r["not_covered_caught"]), len(r["universal_caught"]))
        if key != prev or r["tau"] == chosen:
            mark = "  <- chosen (highest tau with 0 false refusals)" if r["tau"] == chosen else ""
            print(f"{r['tau']:>5.2f} | {key[0]:>14} | {key[1]:>27} | {key[2]:>20}{mark}")
            prev = key
    print(f"\nLowest answerable best_score: {lowest_answerable[0]:.3f} ({lowest_answerable[1]})")
    if highest_off_topic:
        print(
            "Highest universal-silence best_score:"
            f" {highest_off_topic[0]:.3f} ({highest_off_topic[1]})"
        )
    print("NOTE: tau is tuned on the same set it is scored on (no held-out set).")

    result = {
        "chosen_tau": chosen,
        "rule": "highest tau with zero false refusals of answerable cases",
        "caveat": "tuned on the same set it is scored on (no held-out set)",
        "best_scores": dict(sorted(scores.items(), key=lambda kv: kv[1])),
        "lowest_answerable": {"id": lowest_answerable[1], "score": lowest_answerable[0]},
        "sweep": rows,
    }
    out = EVAL_DIR / f"tau_sweep{variant_suffix(embedding=settings.embedding_model)}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Wrote {out}")
    return result


# ---------------------------------------------------------------- full eval


# ---------------------------------------------------------------- retrieval ranks (no LLM)


def expected_rank(case: dict, chunks) -> int | None:
    """1-based rank of the first pooled chunk from an expected section of the right version."""
    want = set(case["expected"]["sections"])
    for i, ch in enumerate(chunks, 1):
        parsed = parse_chunk_id(ch.chunk_id)
        if parsed and parsed[1] == case["expected"]["version"] and parsed[2] in want:
            return i
    return None


def run_ranks(settings: Settings, cases: list[dict], out_name: str | None = None):
    """Where does the expected section land in the retrieval pool? hit@3 (sent), @5, @8 (pool).
    Also flags answerable cases whose section is in the top 3 but tau (vector score) refuses."""
    mode = settings.retrieval_mode
    suffix = variant_suffix(mode, settings.embedding_model)
    out_name = out_name or f"retrieval_ranks{suffix}.json"
    env = Env(settings)
    ranks: dict[str, int | None] = {}
    best_vector: dict[str, float | None] = {}
    try:
        for phase in PHASES:
            if phase == "after_nimbus_v2":
                env.ingest_update()
            for c in cases:
                if c["input"]["phase"] != phase or c["expected"]["status"] != "answered":
                    continue
                res = search(
                    c["input"]["company"],
                    c["input"]["question"],
                    env.registry,
                    env.store,
                    settings.retrieval_pool,
                    mode=mode,
                    bm25=env.bm25,
                    rrf_k=settings.rrf_k,
                )
                ranks[c["id"]] = expected_rank(c, res.chunks)
                best_vector[c["id"]] = res.best_vector_score
    finally:
        env.close()
    n = len(ranks)
    hits = {
        f"hit@{k}": ratio(sum(r is not None and r <= k for r in ranks.values()), n)
        for k in (3, 5, settings.retrieval_pool)
    }
    tau = settings.retrieval_min_score
    found_but_tau = sorted(
        cid
        for cid, r in ranks.items()
        if r is not None and r <= settings.retrieval_top_k and (best_vector[cid] or 0.0) < tau
    )
    out = {
        "mode": mode,
        "embedding_model": settings.embedding_model,
        "pool": settings.retrieval_pool,
        "answerable": n,
        **hits,
        "found_in_top3_but_tau_refuses": found_but_tau,
        "ranks": ranks,
        "best_vector_score": {k: v and round(v, 4) for k, v in best_vector.items()},
    }
    for cid, r in ranks.items():
        if r is None or r > settings.retrieval_top_k:
            print(f"  {cid:<34} rank {r if r else f'> {settings.retrieval_pool} (not in pool)'}")
    print("  " + " | ".join(f"{k} {v}" for k, v in hits.items()) + f"  (n={n})")
    print(f"  found in top 3 but tau ({tau}) refuses: {found_but_tau or 'none'}")
    (EVAL_DIR / out_name).write_text(json.dumps(out, indent=2) + "\n")
    return out


def ask_case(env: Env, llm: LLMClient, case: dict, throttle_s: float) -> tuple[dict, list[str]]:
    """Run one case through the real answerer. Retries only provider rate limits (eval-only)."""
    notes: list[str] = []
    p = Principal(sub="eval", company_id=case["input"]["company"], role="employee")
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        resp, trace = answer_question(
            p, case["input"]["question"], env.registry, env.store, llm, env.settings, bm25=env.bm25
        )
        if trace.error == "rate_limited" and attempt < RATE_LIMIT_RETRIES:
            notes.append(f"rate-limited, retry {attempt + 1} after {RATE_LIMIT_WAIT_S:.0f}s")
            time.sleep(RATE_LIMIT_WAIT_S)
            continue
        break
    if resp.meta.llm_calls and throttle_s:
        time.sleep(throttle_s)
    actual = {
        "status": resp.status.value,
        "answer": resp.answer,
        "citations": [c.chunk_id for c in resp.citations],
        "gate": trace.gate,
        "best_score": None if trace.best_score is None else round(trace.best_score, 4),
        "retrieved_ids": trace.retrieved_ids[:3],
        "llm_calls": resp.meta.llm_calls,
        "latency_ms": resp.meta.latency_ms,
        "policy_version": resp.meta.policy_version,
    }
    if trace.retried:
        notes.append(f"retried={trace.retried}")  # D53 connection-level retry (counted)
    if trace.error:
        notes.append(f"error={trace.error}")
    return actual, notes


def run_once(settings: Settings, cases: list[dict], llm: LLMClient, throttle_s: float):
    env = Env(settings)
    results = []
    try:
        for phase in PHASES:
            if phase == "after_nimbus_v2":
                env.ingest_update()
            for case in (c for c in cases if c["input"]["phase"] == phase):
                actual, extra = ask_case(env, llm, case, throttle_s)
                score = score_case(case, actual)
                results.append((case, actual, score, extra))
                mark = "PASS" if score.passed else "FAIL"
                print(
                    f"  {mark}  {case['id']:<34} {actual['status']:<12} gate={actual['gate']:<16}"
                    f" best={actual['best_score']} calls={actual['llm_calls']}"
                    f" {actual['latency_ms']}ms"
                    + ("" if score.passed else f"  <- {'; '.join(score.failures)}")
                )
    finally:
        env.close()
    return results


def ratio(num: int, den: int) -> float | None:
    return round(num / den, 3) if den else None


def build_report(results, run_info: dict) -> dict:
    total = len(results)
    passed = sum(s.passed for _, _, s, _ in results)
    by_cat: dict[str, Counter] = defaultdict(Counter)
    for case, _, s, _ in results:
        cat = case["notes"].split(";")[0].replace("category:", "").strip()
        by_cat[cat]["total"] += 1
        by_cat[cat]["pass"] += s.passed

    ans = [(c, a, s) for c, a, s, _ in results if c["expected"]["status"] == "answered"]
    ans_answered = [(c, a, s) for c, a, s in ans if a["status"] == "answered"]
    pred_nc = [(c, a) for c, a, _, _ in results if a["status"] == "not_covered"]
    exp_nc = [(c, a) for c, a, _, _ in results if c["expected"]["status"] == "not_covered"]
    gated = [(c, a, s) for c, a, s, _ in results if "gate" in c["expected"]]
    llm_lat = [a["latency_ms"] for _, a, _, _ in results if a["llm_calls"] > 0]
    all_lat = [a["latency_ms"] for _, a, _, _ in results]

    return {
        "run": run_info,
        "summary": {"total": total, "passed": passed, "pass_rate": ratio(passed, total)},
        "by_category": {k: dict(v) for k, v in sorted(by_cat.items())},
        "metrics": {
            "retrieval_hit_at_3": ratio(sum(bool(s.hit_at_3) for _, _, s in ans), len(ans)),
            "citation_accuracy": ratio(
                sum(bool(s.sections_ok) for _, _, s in ans_answered), len(ans_answered)
            ),
            "key_fact_accuracy": ratio(
                sum(bool(s.facts_ok) for _, _, s in ans_answered), len(ans_answered)
            ),
            "not_covered_precision": ratio(
                sum(c["expected"]["status"] == "not_covered" for c, _ in pred_nc), len(pred_nc)
            ),
            "not_covered_recall": ratio(
                sum(a["status"] == "not_covered" for _, a in exp_nc), len(exp_nc)
            ),
            "not_covered_by_gate": dict(Counter(a["gate"] for _, a in pred_nc)),
            "gate_as_expected": ratio(sum(bool(s.gate_as_expected) for *_, s in gated), len(gated)),
            "false_refusals": sorted(c["id"] for c, a, _ in ans if a["status"] == "not_covered"),
            "leaks": sum(s.leak for _, _, s, _ in results),
            "injection_passed": f"{by_cat['injection']['pass']}/{by_cat['injection']['total']}",
            "try_again": sum(a["status"] == "try_again" for _, a, _, _ in results),
            "connection_retries": sum(
                any(n.startswith("retried=") for n in x) for *_, x in results
            ),
            "max_llm_calls": max(a["llm_calls"] for _, a, _, _ in results),
            "total_llm_calls": sum(a["llm_calls"] for _, a, _, _ in results),
            "max_citations": max(len(a["citations"]) for _, a, _, _ in results),
            "latency_ms": {
                "note": "in-process (retrieval + LLM), excludes HTTP; see scripts/bench.py for API",
                "llm_cases_p50": percentile(llm_lat, 50),
                "llm_cases_p95": percentile(llm_lat, 95),
                "all_cases_p95": percentile(all_lat, 95),
                "max": max(all_lat) if all_lat else None,
            },
        },
        "failures": {c["id"]: s.failures for c, _, s, _ in results if not s.passed},
    }


def to_row(case: dict, actual: dict, score, extra: list[str], provider: str, model: str) -> dict:
    exp_gate = case["expected"].get("gate")
    note_parts = [
        case["notes"],
        f"provider={provider}"
        + (f" model={model}" if provider == "groq" else " (extractive fake)"),
        f"gate={actual['gate']}" + (f" (expected {exp_gate})" if exp_gate else ""),
        f"best_score={actual['best_score']}",
        f"llm_calls={actual['llm_calls']}",
        *extra,
    ]
    if not score.passed:
        note_parts.append("FAIL: " + "; ".join(score.failures))
    return {
        "id": case["id"],
        "input": case["input"],
        "expected": case["expected"],
        "actual": actual,
        "pass": score.passed,
        "notes": "; ".join(note_parts),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--llm", choices=["auto", "groq", "fake"], default="auto")
    ap.add_argument("--sweep", action="store_true", help="tau sweep, retrieval only (no LLM)")
    ap.add_argument("--ranks", action="store_true", help="expected-section rank in the pool")
    ap.add_argument("--retrieval", choices=["vector", "hybrid"], help="override RETRIEVAL_MODE")
    ap.add_argument("--embedding", choices=["minilm", "bge-small"], help="override EMBEDDING_MODEL")
    ap.add_argument("--tau", type=float, help="override RETRIEVAL_MIN_SCORE for this run")
    ap.add_argument("--repeat", type=int, default=1, help="run N times and report flips")
    ap.add_argument("--throttle", type=float, help="seconds between LLM cases (groq default 2.5)")
    args = ap.parse_args(argv)

    settings = get_settings()
    if args.tau is not None:
        settings = settings.model_copy(update={"retrieval_min_score": args.tau})
    if args.retrieval:
        settings = settings.model_copy(update={"retrieval_mode": args.retrieval})
    if args.embedding:
        settings = settings.model_copy(update={"embedding_model": args.embedding})
    cases = load_cases()

    if args.ranks:
        banner(
            [
                f"RETRIEVAL RANKS - no LLM | {settings.retrieval_mode}"
                f" | {settings.embedding_model}"
                f" | pool {settings.retrieval_pool}"
            ]
        )
        run_ranks(settings, cases)
        return 0
    if args.sweep:
        banner([f"TAU SWEEP (retrieval only, no LLM) - {len(cases)} cases"])
        run_sweep(settings, cases)
        return 0

    provider = choose_provider(args.llm, settings)
    if provider == "groq" and not settings.groq_api_key:
        sys.exit("--llm groq needs GROQ_API_KEY in .env")
    settings = settings.model_copy(update={"llm_provider": provider})
    llm = make_llm(settings)
    throttle = args.throttle if args.throttle is not None else (2.5 if provider == "groq" else 0.0)
    results_path, report_path = output_paths(
        provider, settings.retrieval_mode, settings.embedding_model
    )

    banner(
        [
            f"EVAL - provider: {provider.upper()}"
            + (
                f" ({settings.groq_model})"
                if provider == "groq"
                else " (extractive fake, NO real LLM)"
            ),
            f"cases: {len(cases)} | tau: {settings.retrieval_min_score} | top_k: "
            f"{settings.retrieval_top_k} | pool: {settings.retrieval_pool} | throttle: {throttle}s",
            f"writes: {results_path.relative_to(EVAL_DIR.parent)}"
            + ("" if provider == "groq" else "  (fake runs never write eval/results.jsonl)"),
        ]
    )

    runs = []
    for i in range(args.repeat):
        if args.repeat > 1:
            print(f"\n--- run {i + 1}/{args.repeat}")
        runs.append(run_once(settings, cases, llm, throttle))

    results = runs[0]
    run_info = {
        "provider": provider,
        "model": settings.groq_model if provider == "groq" else "extractive-fake",
        "retrieval_mode": settings.retrieval_mode,
        "tau": settings.retrieval_min_score,
        "top_k": settings.retrieval_top_k,
        "pool": settings.retrieval_pool,
        "cases": len(cases),
        "commit": git_commit(),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "repeats": args.repeat,
    }
    report = build_report(results, run_info)
    if args.repeat > 1:
        flips = {}
        for case_i, (case, *_rest) in enumerate(results):
            outcomes = [run[case_i][2].passed for run in runs]
            if len(set(outcomes)) > 1:
                flips[case["id"]] = outcomes
        report["flips_across_repeats"] = flips

    model = settings.groq_model
    with results_path.open("w") as f:
        for case, actual, score, extra in results:
            f.write(
                json.dumps(to_row(case, actual, score, extra, provider, model), ensure_ascii=False)
                + "\n"
            )
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    m, sm = report["metrics"], report["summary"]
    banner(
        [
            f"{provider.upper()}: {sm['passed']}/{sm['total']} passed ({sm['pass_rate']:.0%})",
            f"hit@3 {m['retrieval_hit_at_3']} | citation acc {m['citation_accuracy']} | "
            f"key facts {m['key_fact_accuracy']}",
            f"not-covered P/R {m['not_covered_precision']}/{m['not_covered_recall']} "
            f"by gate {m['not_covered_by_gate']} | leaks {m['leaks']}",
            f"max LLM calls {m['max_llm_calls']} | max citations {m['max_citations']} | "
            "LLM-case p50/p95 "
            f"{m['latency_ms']['llm_cases_p50']}/{m['latency_ms']['llm_cases_p95']} ms",
            *(
                [f"flips across {args.repeat} runs: {report['flips_across_repeats'] or 'none'}"]
                if args.repeat > 1
                else []
            ),
            f"wrote {results_path.name} + {report_path.name}",
        ]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
