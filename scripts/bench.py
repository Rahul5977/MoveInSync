"""Latency benchmark against the RUNNING API (HTTP included), e.g. after `docker compose up`.

  uv run python -m scripts.bench --n 30 --throttle 2.5

Cycles through the answerable `seed`-phase questions in eval/cases.jsonl (so each request
normally makes one real LLM call), throttled to respect Groq free-tier rate limits. Reports
p50/p95 of client wall-clock time and of the server's own `meta.latency_ms`, plus status counts.
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
    wall, server, statuses, calls = [], [], Counter(), Counter()
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
            calls[body["meta"]["llm_calls"]] += 1
            print(f"  {i + 1:>3} {co:<8} {body['status']:<12} {ms:7.0f} ms  {q['question'][:60]}")
            if i < args.n - 1:
                time.sleep(args.throttle)

    out = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "base": args.base,
        "health": health,
        "n": args.n,
        "throttle_s": args.throttle,
        "client_ms": {
            "p50": round(percentile(wall, 50)),
            "p95": round(percentile(wall, 95)),
            "max": round(max(wall)),
        },
        "server_ms": {"p50": percentile(server, 50), "p95": percentile(server, 95)},
        "statuses": dict(statuses),
        "llm_calls_histogram": {str(k): v for k, v in sorted(calls.items())},
        "under_3s_p95": percentile(wall, 95) < 3000,
    }
    (EVAL_DIR / "bench.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
