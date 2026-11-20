# Evaluation Report — Commute Policy Assistant

Status: **draft** (Fri 20 Nov). Sections marked _TBD_ are filled from `eval/report.json`,
`eval/tau_sweep.json` and `eval/bench.json` once the runs are done.

## 1. What is evaluated
One fixed set of questions (`eval/cases.jsonl`, 31 cases), scored the same way on every run by
`eval/run_eval.py`. Results go to `eval/results.jsonl` with exactly the fields
`id, input, expected, actual, pass, notes`.

| Category | Cases | What it checks |
|---|---|---|
| direct | 6 | One clear rule per question, at least one per company |
| paraphrase | 3 | Informal wording, abbreviation ("sat"), no punctuation |
| policy-says-no | 2 | A rule that says "no" is still *covered* (Kestrel Saturday, Saffron vendor) |
| distractor | 4 | Neighbouring sections: night drops vs escort, airport approval vs cost, ad-hoc approval vs cap, no-show vs wait time |
| cross-company | 3 | Questions that name or hint at another company's rule |
| asymmetric-silence | 3 | Topics another company covers but this one doesn't → must be "not covered" (LLM gate) |
| universal-silence | 3 | Topics no policy covers (pets, Wi-Fi, children) → "not covered" (τ gate, 0 LLM calls) |
| injection | 2 | Planted instructions in a policy (Saffron §10) and in the question |
| versioning | 4 | Same questions before and after Nimbus v2 is ingested (escort 8:00 → 7:30 PM, cut-off 4 → 3 h) |
| multi-section | 1 | Answer needs two sections (Saffron §10 approval + §11 cap) |

## 2. Method
- **Expected answers come from the policy text only**, never from running the system. Each case
  names the section(s) that support it.
- **Fresh environment per run**: a temporary data dir is seeded with v1 of every company, the
  `seed` cases run, Nimbus v2 is ingested through the real ingestion path, then the
  `after_nimbus_v2` cases run. No local state leaks in.
- **Matching** (D48): answer and phrases are normalised (lowercase, commas stripped, ₹/Rs/INR
  unified, "p.m." → "pm") and matched on word boundaries. `must_include` is a list of any-of groups
  (e.g. `["8:00 PM", "8 PM", "20:00"]`), so correct answers aren't failed for phrasing.
- **Pass rule** (D49): status matches; if answered, the cited section(s) are the expected ones
  from the expected version; every `must_include` group present; no `must_not_include` phrase;
  zero chunks from another company retrieved or cited. Which gate said "not covered" (τ vs LLM) is
  reported separately, not pass-critical.
- **Providers** (D38e): with no `GROQ_API_KEY` the runner uses an extractive fake and writes
  `eval/results_fake.jsonl`; only a real Groq run writes `eval/results.jsonl`. The provider is in
  every row's `notes` and in the report.

## 3. Results
_TBD: Groq run (pass rate, by category), fake run, metrics: retrieval hit@3, citation accuracy,
key-fact accuracy, not-covered precision/recall by gate, leaks (must be 0), max LLM calls,
max citations._

## 4. Threshold τ
_TBD: sweep table (false refusals vs not-covered caught per τ), chosen value._
τ is tuned on the same 31 cases it is scored on; there is no held-out set, so the reported
not-covered numbers for the τ gate are optimistic.

## 5. Latency
_TBD: `scripts/bench.py`, ≥ 30 real Groq requests against the running API, p50/p95._

## 6. Known failures
_TBD from the runs._

## 7. Incidents

### 7.1 Groq retired the configured model (Fri 20 Nov, ~13:40 IST)
**What failed.** The first real-Groq smoke test returned HTTP 404 on every LLM call:
`The model 'llama-3.1-8b-instant' does not exist or you do not have access to it`
(`model_not_found`). The model we had designed around (D2) had been retired by the provider.
`GET /v1/models` on the same key listed no Llama chat models at all.

**How the system behaved.** As designed, and safely:
- Every question that reached the LLM got a clean **HTTP 503 `try_again`** with the fixed
  "please try again or contact the transport desk" text: no 500, no hang, no half answer
  (provider errors → `LLMError` → `try_again`, D38/D41).
- The off-topic question ("Can I bring my pet dog in the cab?") was still refused correctly by the
  **τ gate with 0 LLM calls**, because that path never depends on the model.
- `/health` stayed 200, and the admin ingest of Nimbus v2 still worked: only answering was degraded.

**How it was fixed (D45).** We probed the models still available on the key with our real prompt
(an answerable "no" question and a not-covered question):

| Model | Escort at 7:45 PM (answerable "no") | Kestrel guest (not covered) | Latency |
|---|---|---|---|
| gpt-oss-20b (defaults) | HTTP 400 `json_validate_failed` (reasoning used the 300-token budget) | ✅ | 937 ms |
| gpt-oss-20b (`reasoning_effort=low`) | ❌ marked `covered=false` | ✅ | ~420 ms |
| gpt-oss-120b (`reasoning_effort=low`) | ✅ | ✅ | 490–870 ms |
| **qwen/qwen3.8-27b** | ✅ | ✅ | **~320 ms** |

We switched the default to `qwen/qwen3.8-27b` (one setting; no code change in the answer path),
and the rerun smoke test passed 8/8. The incident also produced two hardening changes:
- **D46**: Groq's `400 json_validate_failed` is now treated as invalid model output (one repair
  call if time allows) instead of a provider failure.
- **D47**: the system prompt now states that a rule the policy states is "covered" even when the
  answer is "no" (gpt-oss-20b's mistake).

**Lesson.** A hosted free-tier model can disappear without notice. The design held up because the
safety-critical decisions don't depend on the model, and the model is one config value. A
production version would add a startup probe that checks the configured model exists, so
`/health` reports `ready: false` instead of every request failing.

## 8. Limitations
_TBD._

## 9. Trade-offs
_TBD (at least one, with reasoning)._
