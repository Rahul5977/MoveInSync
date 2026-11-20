# Commute Policy Assistant

Answers employee transport-policy questions from **their own company's current policy only**,
cites document + section, and says plainly when the policy is silent.

> Status: scaffolding. Design decisions and trade-offs will be filled in as they are finalised.

## Run (one command)
```bash
cp .env.example .env   # add GROQ_API_KEY for the live demo
docker compose up
```
API at http://localhost:8000 — docs at http://localhost:8000/docs.

## Local development
```bash
uv sync
uv run pytest                                              # tests (fake LLM, no network)
uv run python -m scripts.issue_token --company acme --sub emp1
uv run uvicorn app.main:app --reload
uv run python -m eval.run_eval                             # writes eval/results.jsonl
```

## Design decisions
_TBD_

## Trade-offs
_TBD_

## Staying within the budget (≤ 2 LLM calls, ≤ 3 citations, p95 < 3 s)
_TBD_
