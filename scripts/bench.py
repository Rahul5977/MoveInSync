"""Latency benchmark against the RUNNING API (HTTP included), e.g. after `docker compose up`.

  uv run python -m scripts.bench --n 30 --throttle 2.5

Cycles through the answerable `seed`-phase questions in eval/cases.jsonl (so each request
normally makes one real LLM call), throttled to respect Groq free-tier rate limits. Reports
p50/p95 of client wall-clock time and of the server's own `meta.latency_ms`, plus status counts.
Answered requests and try_again responses (e.g. Groq 429) are also reported separately: the
first is the latency budget, the second is how fast a failure reaches the user.
Writes eval/bench.json. Run it on a server with Nimbus at v1 (questions are seed-phase).
"""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import httpx

from app.auth import issue_token
from app.config import get_settings
from eval.scoring import percentile

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--throttle", type=float, default=2.5, help="seconds between requests")
    ap.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="untimed off-topic requests first (refused by tau: 0 LLM calls, 0 tokens)",
    )
    ap.add_argument("--label", default="", help="free text stored in the output, e.g. the mode")
    ap.add_argument("--out", default="bench.json", help="file name under eval/")
    args = ap.parse_args(argv)

    settings = get_settings()
    cases = [json.loads(x) for x in (EVAL_DIR / "cases.jsonl").read_text().splitlines() if x]
    pool = [
        c["input"]
        for c in cases
        if c["input"]["phase"] == "seed" and c["expected"]["status"] == "answered"
    ]
    health = httpx.get(f"{args.base}/health", timeout=5).json()
    print(
        f"health: {health}  |  {args.n} requests, throttle {args.throttle}s, {len(pool)} questions"
    )
    if not health.get("llm", {}).get("ready"):
        print("WARNING: server LLM not ready; latencies will reflect try_again responses")

    tokens = {}
    with httpx.Client(timeout=10) as client:  # warm the embedding model / BM25, not counted
        for i in range(args.warmup):
            co = pool[i % len(pool)]["company"]
            tokens.setdefault(co, issue_token("bench", co, "employee", settings))
            client.post(
                f"{args.base}/ask",
                json={"question": "Can I bring my pet dog in the cab?"},
                headers={"Authorization": f"Bearer {tokens[co]}"},
            )
    wall, server, statuses, calls = [], [], Counter(), Counter()
    by_status: dict[str, list[float]] = {}
    answered_by_calls: dict[int, list[float]] = {}  # D61: 1-call vs 2-call answers
    with httpx.Client(timeout=10) as client:
        for i in range(args.n):
            q = pool[i % len(pool)]
            co = q["company"]
            if co not in tokens:
                tokens[co] = issue_token("bench", co, "employee", settings)
            t0 = time.perf_counter()
            r = client.post(
                f"{args.base}/ask",
                json={"question": q["question"]},
                headers={"Authorization": f"Bearer {tokens[co]}"},
            )
            ms = (time.perf_counter() - t0) * 1000
            body = r.json()
            wall.append(ms)
            server.append(body["meta"]["latency_ms"])
            statuses[body["status"]] += 1
            by_status.setdefault(body["status"], []).append(ms)
            calls[body["meta"]["llm_calls"]] += 1
            if body["status"] == "answered":
                answered_by_calls.setdefault(body["meta"]["llm_calls"], []).append(ms)
            print(f"  {i + 1:>3} {co:<8} {body['status']:<12} {ms:7.0f} ms  {q['question'][:60]}")
            if i < args.n - 1:
                time.sleep(args.throttle)

    out = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "base": args.base,
        "health": health,
        "n": args.n,
        "throttle_s": args.throttle,
        "warmup_requests": args.warmup,
        "label": args.label,
        "client_ms": {
            "p50": round(percentile(wall, 50)),
            "p95": round(percentile(wall, 95)),
            "max": round(max(wall)),
        },
        "server_ms": {"p50": percentile(server, 50), "p95": percentile(server, 95)},
        "statuses": dict(statuses),
        "client_ms_by_status": {
            st: {
                "n": len(v),
                "p50": round(percentile(v, 50)),
                "p95": round(percentile(v, 95)),
                "max": round(max(v)),
            }
            for st, v in sorted(by_status.items())
        },
        "llm_calls_histogram": {str(k): v for k, v in sorted(calls.items())},
        "answered_ms_by_llm_calls": {
            str(k): {
                "n": len(v),
                "p50": round(percentile(v, 50)),
                "p95": round(percentile(v, 95)),
                "max": round(max(v)),
            }
            for k, v in sorted(answered_by_calls.items())
        },
        "under_3s_p95": percentile(wall, 95) < 3000,
    }
    (EVAL_DIR / args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
