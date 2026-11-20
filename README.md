# Commute Policy Assistant

Employees ask transport-policy questions ("Do I get an escort for a 7:45 PM drop?") and get an
answer **only from their own company's current policy**, with the document and section cited. When
the policy is silent, the assistant says so instead of guessing. Three demo companies (Nimbus
Software, Kestrel Bank, Saffron Pharma) share one deployment and can never see each other's rules.

Stack: Python 3.12, FastAPI, Chroma, local ONNX MiniLM embeddings, Groq LLM, JWT.

**Demo video (≈ 2 min):** [Watch on Google Drive](https://drive.google.com/file/d/1ETWSByFX8reqsg2U8GiBUcVpTQwQ9ezU/view?usp=sharing)<br>
Rendered with VHS from `scripts/demo.tape` against the live Docker stack (real Groq answers);
reproducible with `vhs scripts/demo.tape`. The same steps run interactively with `scripts/demo.sh`.

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

## For reviewers: where to look

Each claim points to the evidence (tests in `tests/`, sections in [docs/EVAL_REPORT.md](docs/EVAL_REPORT.md)).

- **Cross-company isolation.** Tests `test_cross_company_retrieval_never_leaks`,
  `test_cross_company_question_sees_only_own_policy`, `test_client_cannot_override_company`,
  `test_admin_cannot_ingest_another_companys_document`, `test_hybrid_isolation_no_foreign_chunks`.
  Eval: **0 leaks** across 35 cases, including 3 cross-company traps (`kes-asks-nimbus-escort`,
  `nim-saffron-cap-trap`, `nim-escort-10pm-trap`), EVAL_REPORT §3.
- **Atomic versioning.** `test_failed_ingest_leaves_old_version` (forced mid-ingest failure),
  `test_crash_mid_ingest_reconciled_on_restart`, `test_restart_keeps_v2`,
  `test_ask_flips_after_v2_ingest`; the v1 → v2 switch is steps 8–10 of the demo video.
- **Model output is untrusted.** Strict JSON with one repair, then `try_again`
  (`test_bad_json_then_repair_succeeds`, `test_bad_json_twice_is_try_again`); only cited ids we
  sent are kept (`test_unknown_and_foreign_citations_dropped`,
  `test_only_invalid_citations_becomes_not_covered`, `test_citations_deduplicated_and_capped_at_three`);
  planted instructions stay data (`test_document_injection_stays_inside_data_block`).
- **A real incident handled.** Groq retired `llama-3.1-8b-instant` mid-build; the system degraded
  to `try_again` as designed and was fixed with one setting (EVAL_REPORT §7.1).
- **Honest evaluation.** τ tuned on the same cases, stated with its 0.017 margin (§4); every
  answer reviewed by hand, 0 disagreements (§10); 2 held-out injection cases added before the
  prompt change ran (§3.1); the default retrieval mode rule fixed before measuring, and the BM25
  regression (`nim-weekend-typo` 4 → 8) reported, so vector stays the default (§3.3). A 12-question
  **held-out set** written after tuning (§3.5) exposed τ refusing 4 of 10 answerable questions; τ
  was deliberately not re-tuned on it (D62). Four experiments were pre-registered with merge rules
  and reported pass or fail (§3.3, §3.4, §3.6, §3.7); only D61 passed.
- **Known failures: 2/35 and 5/12 held-out, all safe refusals except one scorer miss** ("not
  covered", never a wrong or foreign rule): vocabulary mismatch on the 35 (§6), τ on the held-out (§3.5).

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
docker compose --profile test run --rm tests                                  # 172 tests, keyless
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

From [docs/EVAL_REPORT.md](docs/EVAL_REPORT.md) (Groq `qwen/qwen3.8-27b`, rewrite on a miss enabled, D61):

| | |
|---|---|
| Pass rate, 35 cases | **33/35** (32/35 before D61; both failures are safe "not covered" refusals from retrieval misses) |
| Pass rate, 12 held-out | **7/12**, written after tuning. τ refuses 4 answerable ones before the LLM (§3.5, D62); 1 is a scorer miss |
| Cross-company leaks | **0** (35 + held-out) |
| Not-covered recall | **1.00** (every question the policy doesn't cover is refused, 9/9) |
| Citation / key-fact accuracy | 1.00 / 1.00 on answered cases (35) |
| Latency (Docker, HTTP, real Groq) | answered p95 **867 ms**; a 2-call answer 1,017 ms; budget 3 s |
| LLM calls | 1 for most questions; 2 for a repair, a dropped connection, or a round-2 search after a miss; never 3 |

The author reviews every answer by hand (§10); auto-scores are never changed afterwards.

## 6. Design decisions (summary)

Full log with 62 decisions and their reasons: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

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
- **Rewrite on a miss (D61):** if the model says "not covered" it may suggest a search phrase.
  One more search (same company, same active policy) and one more call answering the **original**
  question happen only if the first answer used exactly 1 call, ≥ 1.2 s is left and the new top 3
  sections differ. So a repair, a retry or a rewrite, never two of them: at most 2 calls. Any
  failure in round 2 → "not covered" (round 1 already said so). The phrase is never shown to the
  user or put in a prompt.
- **≤ 3 citations by construction:** only the top 3 sections are sent, and citations must be among them.

## 8. Trade-offs and known limitations

- **τ is overfitted to the 35 cases.** τ = 0.38 was tuned on them (margin 0.017), and on 12 new
  held-out questions it wrongly refused 4 of 10 answerable ones before any LLM call (scores
  0.324–0.377). It was not re-tuned on the held-out set (that would leave no unbiased test). Small,
  self-written eval sets (35 + 12, one author).
- **Retrieval misses from vocabulary mismatch** ("fined" vs "deducted", "in advance" vs "cut-off",
  "sat" vs "Saturday"). They fail safe ("not covered"). The hybrid BM25 + RRF bonus is built
  (`RETRIEVAL_MODE=hybrid`) but **vector stays the default**: hit@3 didn't improve and one case got
  worse. A stronger embedder (bge-small, D59) was also measured and **not merged**: hit@3 0.893 →
  0.929 but one case got worse, and its scores no longer separate off-topic questions (τ margin
  +0.017 → −0.039). A cross-encoder reranker (D60) fixed some misses but lost one and added
  ~0.6 s: **not merged**. Rewrite on a miss (D61) fixed "sat" and one held-out question: **merged**.
- **Cross-company questions are refused**, not answered with the user's own rule. Less helpful,
  but it can never be mistaken for another company's rule.
- **Fixed refusal text** hides the model's reason for refusing, so a refusal can never contain a guess.
- **No semantic judge** in the eval: correctness = right section + required phrases, plus a manual review.
- **Free-tier Groq:** 8,000 tokens/minute and 200,000/day (~130 questions a day). Beyond that,
  users get `try_again`. A hosted model can be retired without notice (it happened; fixed by one setting).
- **English-only embeddings** (MiniLM, 256 tokens); **single process** (the ingest lock is per process).

## 9. What I'd build next

Each item: what it fixes, and what it costs.

- **Recalibrate τ** on a larger labelled set with its own held-out split, or make τ a coarse gate
  (only clearly off-topic, e.g. < 0.3) and let the LLM's "not covered" decide the rest. Fixes the
  4 held-out false refusals (D62); costs one call per near-miss question.
- **A bigger held-out set** (50+ questions, ideally written by someone else): the 35 + 12 are now
  both used for decisions, so the next change needs fresh questions.
- **A cheaper reranker.** The cross-encoder (D60) reached sections the vector top 8 missed, but
  scoring all ~22 chunks cost ~0.5 s and lost one case. Rerank only the vector top 8 plus a few
  BM25 hits, or use a smaller model; judge it on new held-out questions.
- **Stronger embedder.** Needs a recalibrated or relative not-covered gate first: with bge-small
  the off-topic and answerable scores overlap (EVAL_REPORT §3.4).
- **Scale.** Postgres with a database-level ingest lock (several workers); background ingestion with
  a job status endpoint; PDF/DOCX parsing; an answer cache keyed by company + active version +
  question, so a new policy version never serves a stale answer.

## 10. Repo layout

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
tests/               172 tests, keyless
docs/                ARCHITECTURE.md (design + decision log), EVAL_REPORT.md
```
