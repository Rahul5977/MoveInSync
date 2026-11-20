# Commute Policy Assistant

Employees ask transport-policy questions ("Do I get an escort for a 7:45 PM drop?") and get an
answer **only from their own company's current policy**, with the document and section cited. When
the policy is silent, the assistant says so instead of guessing. Three demo companies (Nimbus
Software, Kestrel Bank, Saffron Pharma) share one deployment and can never see each other's rules.

Stack: Python 3.12, FastAPI, Chroma, local ONNX MiniLM embeddings, Groq LLM, JWT.

## 1. Quick start (one command)

```bash
docker compose up --build
```

The API is at http://localhost:8000 (interactive docs at `/docs`). On first start it loads v1 of all
three policies from `data/policies/seed/`.

- **No key is needed to start.** Without one, `GET /health` shows `"ready": false` and `/ask`
  returns `try_again` (HTTP 503) for questions that need the LLM (D38). Off-topic questions are
  still refused correctly, with no LLM call.
- **For real answers:** `cp .env.example .env`, put a free Groq key (https://console.groq.com) in
  `GROQ_API_KEY`, and restart. The model is one setting, `GROQ_MODEL` (default
  `qwen/qwen3.8-27b`); if Groq retires it, as happened to our first model, set another one there
  ([incident, D45](docs/EVAL_REPORT.md#71-groq-retired-the-configured-model-fri-20-nov-1340-ist)).
- **Reset all state** (back to v1 everywhere): `docker compose down -v`.

## 2. Tokens

A dev token issuer stands in for the company's identity provider. The company comes **only** from
the token. Companies: `nimbus`, `kestrel`, `saffron`. Roles: `employee` (default) and `admin`.

```bash
# With Docker only (no Python needed on the host)
TOKEN=$(docker compose run --rm api uv run --no-sync python -m scripts.issue_token --company nimbus --sub emp1)
ADMIN=$(docker compose run --rm api uv run --no-sync python -m scripts.issue_token --company nimbus --sub hr1 --role admin)

# Or with uv
uv sync
TOKEN=$(uv run python -m scripts.issue_token --company kestrel --sub emp2)
```

Both read the same `.env` (or the same defaults) as the server, so the tokens are valid.

## 3. Try it with curl

```bash
# Ask a question (employee token)
curl -s localhost:8000/ask -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"question": "Do I get a security escort for a 7:45 PM drop? I am a woman employee."}'

# Upload a new policy version (admin token, own company only): Nimbus v2 moves the escort to 7:30 PM
curl -s -X POST localhost:8000/admin/ingest -H "Authorization: Bearer $ADMIN" \
  -F file=@data/policies/updates/nimbus/v2.md

# Active version and history (admin)
curl -s localhost:8000/admin/policy -H "Authorization: Bearer $ADMIN"
```

Example `/ask` response for a Nimbus employee on v1 (with a Groq key):

```json
{
  "status": "answered",
  "answer": "No, a security escort is not required for a 7:45 PM drop. The policy mandates a trained security escort only for drops departing after 8:00 PM or pickups scheduled before 6:00 AM.",
  "citations": [
    {
      "document": "Nimbus Software — Employee Transport Policy v1",
      "section": "§8 Women's Safety and Security Escort",
      "chunk_id": "nimbus_v1_s8_p0"
    }
  ],
  "meta": {"llm_calls": 1, "latency_ms": 797, "policy_version": 1, "retrieval_mode": "vector"}
}
```

After the v2 upload, the same question returns "Yes … after 7:30 PM" citing `nimbus_v2_s8_p0`.
`status` is one of `answered`, `not_covered` (fixed text: "Your company's transport policy doesn't
cover this…") or `try_again` (HTTP 503, e.g. no key or the LLM provider is down).

## 4. Tests and evaluation

**With Docker only:**
```bash
docker compose --profile test run --rm tests                                  # 138 tests, keyless
docker compose run --rm api uv run --no-sync python -m eval.run_eval          # eval (see below)
```

**With uv:**
```bash
uv sync
uv run pytest                          # tests never call a real LLM (scripted fake)
uv run python -m eval.run_eval         # eval: groq if GROQ_API_KEY is set, else keyless
uv run python -m eval.run_eval --ranks # retrieval-only diagnostic, no LLM
```

- Without a key the eval uses an **extractive fake LLM** and writes `eval/results_fake.jsonl` (inside
  the container when run with Docker). It checks the pipeline, retrieval, isolation and versioning;
  it can't judge answers.
- **`eval/results.jsonl` is the committed real Groq run, the submission file.** Only a Groq run
  writes it. Each row has exactly `id, input, expected, actual, pass, notes`.
- The test suite takes ~25 s with uv and ~100 s in Docker.

## 5. Headline results

From [docs/EVAL_REPORT.md](docs/EVAL_REPORT.md) (35 hand-written cases, Groq `qwen/qwen3.8-27b`):

| | |
|---|---|
| Pass rate | **32/35** (all 3 failures are safe "not covered" refusals caused by retrieval misses) |
| Cross-company leaks | **0** |
| Not-covered recall | **1.00** (every question the policy doesn't cover is refused) |
| Citation / key-fact accuracy | 1.00 / 1.00 on answered cases |
| Latency (Docker, HTTP, real Groq) | p95 **≈ 1 s** (992 ms), budget 3 s |
| LLM calls | 1 per question in normal runs; max 2 (one connection retry, D53); budget 2 |

The author reviewed every answer by hand: 0 disagreements with the automatic score.

## 6. Design decisions (summary)

Full log with ~57 decisions and their reasons: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

- **Company from the JWT only.** No request field can name a company, collection or version;
  extra fields are rejected (422). Unknown company/role → 403.
- **One Chroma collection per company + version** (`nimbus_v2`). Isolation is physical: a query
  can only reach its own company's collection, and a metadata check on the results backs it up.
- **Atomic version switch.** A new version is fully written and verified, then one SQLite
  transaction moves the active pointer. Readers see v1 or v2, never a mix; a failed upload leaves
  v1 live; a restart never rolls back.
- **Section-level chunks** with a context header (`Company | Document v1 | §8 Title`), so each
  citation is an exact section.
- **Two "not covered" gates.** A similarity threshold τ refuses off-topic questions with 0 LLM calls;
  the LLM's `covered=false` catches near misses (a related section that doesn't answer).
- **Citations are validated in code.** Only ids we actually sent count; document and section text
  is rendered from stored metadata, never from model text. Refusal text is fixed in code.
- **Layered prompt-injection defence.** Policy text and the question are escaped data blocks; the
  prompt says to ignore instructions in them; output is strict JSON; and the model never has
  another company's text in its context, so even a hijacked model can't leak it.

## 7. Staying within ≤ 2 LLM calls and < 3 s

- **τ gate first (D5):** off-topic questions are refused before any LLM call.
- **One call, strict JSON (D6):** the normal path is exactly 1 call; a second call happens only to
  repair invalid output, never a third.
- **One request deadline of 2.8 s (D20):** each LLM call gets min(time left, 2.5 s); no call starts
  with < 0.3 s left; the repair call only if ≥ 1.0 s is left. Otherwise → `try_again`, never a slow
  or half answer.
- **Dropped connections (D53):** one retry, only on connection errors, only with ≥ 1.0 s left. It
  uses the repair slot and counts as a call, so the total stays ≤ 2. Rate limits (429) are not
  retried.
- **≤ 3 citations by construction:** only the top 3 sections are sent, and citations must be among them.

## 8. Trade-offs and known limitations

- **Small, self-written eval set** (35 cases, one author); τ = 0.38 was tuned on the same cases
  (margin only 0.017), so unseen phrasings near the boundary may be refused.
- **Retrieval misses from vocabulary mismatch** ("fined" vs "deducted", "in advance" vs "cut-off",
  "sat" vs "Saturday"). They fail safe ("not covered"). The hybrid BM25 + RRF bonus is built
  (`RETRIEVAL_MODE=hybrid`) but **vector stays the default**: hit@3 didn't improve and one case got
  worse. Query rewriting or a stronger embedding model is the real fix.
- **Cross-company questions are refused**, not answered with the user's own rule. Less helpful,
  but it can never be mistaken for another company's rule.
- **Fixed refusal text** hides the model's reason for refusing, so a refusal can never contain a guess.
- **No semantic judge** in the eval: correctness = right section + required phrases, plus a manual review.
- **Free-tier Groq:** 8,000 tokens/minute and 200,000/day (~130 questions a day). Beyond that,
  users get `try_again`. A hosted model can be retired without notice (it happened; fixed by one setting).
- **English-only embeddings** (MiniLM, 256 tokens); **single process** (the ingest lock is per process).

## 9. Repo layout

```
app/
  main.py            FastAPI app: /health, /ask, /admin/ingest, /admin/policy
  auth.py            JWT verification; company and role come only from the token
  rag/               retriever (vector / hybrid), BM25, prompt, answerer (gates, budget, citations)
  llm/               Groq client, fake LLMs for tests and keyless runs
  ingest/            chunker, ingest service (atomic activation, reconciler, seeding)
  store/             SQLite version registry, Chroma vector store
data/policies/       seed/<company>/v1.md (loaded at start), updates/nimbus/v2.md (upload demo)
eval/                cases.jsonl, run_eval.py, scoring.py, results.jsonl (submission), reports
scripts/             issue_token.py, bench.py
tests/               138 tests, keyless
docs/                ARCHITECTURE.md (design + decision log), EVAL_REPORT.md
```
