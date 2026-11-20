#!/usr/bin/env bash
# 2-minute demo of the Commute Policy Assistant, for a screen recording.
#   docker compose down -v && docker compose up --build     (in another terminal)
#   scripts/demo.sh
# Uses ~4 Groq calls per run (steps 2, 4, 5, 10; step 3 is refused with 0 LLM calls).
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="${BASE:-http://localhost:8000}"
ESCORT_Q="Do I get a security escort for a 7:45 PM drop? I'm a woman employee."

bold() { printf '\n\033[1m%s\033[0m\n' "$*"; }
pause() { read -r -p "Enter for next..." _; }
token() { docker compose run --rm -T api uv run --no-sync python -m scripts.issue_token "$@" 2>/dev/null; }

# Show an /ask reply: the fields a viewer cares about.
ASK_VIEW='{status, answer, citations, llm_calls: .meta.llm_calls, latency_ms: .meta.latency_ms}'

# call <jq filter> <curl args...>: print the HTTP status, then the body through jq.
call() {
  local filter="$1"; shift
  local body code
  body=$(mktemp)
  code=$(curl -s -o "$body" -w '%{http_code}' "$@")
  echo "HTTP $code"
  jq "$filter" "$body" 2>/dev/null || cat "$body"
  rm -f "$body"
}

ask() { # ask <token> <question>
  call "$ASK_VIEW" "$BASE/ask" -H "Authorization: Bearer $1" -H "Content-Type: application/json" \
    -d "$(jq -n --arg q "$2" '{question: $q}')"
}

echo "Run \`docker compose down -v && docker compose up --build\` first, so Nimbus starts on v1."

if ! command -v jq >/dev/null; then
  echo "jq is required: brew install jq   (Debian/Ubuntu: sudo apt-get install jq)"
  exit 1
fi
if ! curl -sf "$BASE/health" >/dev/null; then
  echo "API not reachable at $BASE. Start it with: docker compose up --build"
  exit 1
fi

echo "Issuing tokens (dev issuer, via Docker)..."
NIMBUS=$(token --company nimbus --sub emp-nimbus)
KESTREL=$(token --company kestrel --sub emp-kestrel)
SAFFRON=$(token --company saffron --sub emp-saffron)
NIMBUS_ADMIN=$(token --company nimbus --sub hr-nimbus --role admin)

# /health deliberately shows no tenant data (D35), so the version check uses the admin endpoint.
NIMBUS_VERSION=$(curl -s "$BASE/admin/policy" -H "Authorization: Bearer $NIMBUS_ADMIN" \
  | jq -r '.active_version')
if [ "$NIMBUS_VERSION" != "1" ]; then
  echo "Nimbus is on v$NIMBUS_VERSION, not v1. Reset: docker compose down -v && docker compose up --build"
  exit 1
fi
# Warm the embedding model with an off-topic question (refused by tau: no LLM call, no tokens).
curl -s "$BASE/ask" -H "Authorization: Bearer $NIMBUS" -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}' >/dev/null
echo "Nimbus is on v1. Ready."
pause

bold "1. Health: is the API up, and can it answer (LLM ready)?"
call '.' "$BASE/health"
pause

bold "2. Nimbus employee: escort for a 7:45 PM drop? (policy v1: escort after 8:00 PM)"
ask "$NIMBUS" "$ESCORT_Q"
pause

bold "3. Nimbus employee: pet dog in the cab? Not in the policy: refused with 0 LLM calls"
ask "$NIMBUS" "Can I bring my pet dog in the cab?"
pause

bold "4. Kestrel employee asks about Nimbus's rule: refused, Kestrel can't see Nimbus's policy"
ask "$KESTREL" "What time does Nimbus Software require a security escort after?"
pause

bold "5. Saffron employee: ad-hoc approval. Saffron §10 contains a planted instruction; it's ignored"
ask "$SAFFRON" "Who approves an ad-hoc trip?"
pause

bold "6. Kestrel employee tries to switch company in the request body: rejected (422)"
call '.' "$BASE/ask" -H "Authorization: Bearer $KESTREL" -H "Content-Type: application/json" \
  -d '{"question": "Do I get a security escort for a 7:45 PM drop?", "company_id": "nimbus"}'
pause

bold "7. Nimbus employee (not admin) tries to upload a policy: forbidden (403)"
call '.' -X POST "$BASE/admin/ingest" -H "Authorization: Bearer $NIMBUS" \
  -F file=@data/policies/updates/nimbus/v2.md
pause

bold "8. Nimbus admin uploads policy v2 (escort from 7:30 PM): written, then switched atomically (201)"
call '.' -X POST "$BASE/admin/ingest" -H "Authorization: Bearer $NIMBUS_ADMIN" \
  -F file=@data/policies/updates/nimbus/v2.md
pause

bold "9. Nimbus admin: active version is now v2, v1 retired"
call '.' "$BASE/admin/policy" -H "Authorization: Bearer $NIMBUS_ADMIN"
pause

bold "10. Same 7:45 PM question again: now 'yes', citing v2"
ask "$NIMBUS" "$ESCORT_Q"
pause

bold "11. Evaluation headline (eval/report.json: real Groq run, 35 hand-written cases)"
jq -r '"Pass rate: \(.summary.passed)/\(.summary.total) (\(.summary.pass_rate * 100 | round)%)",
  "Cross-company leaks: \(.metrics.leaks)",
  "Not-covered recall: \(.metrics.not_covered_recall)",
  "Max LLM calls per question: \(.metrics.max_llm_calls) (budget <= 2)",
  "p95 latency (in-process, LLM cases): \(.metrics.latency_ms.llm_cases_p95) ms"' eval/report.json
if [ -f eval/bench_groq_vector.json ]; then
  jq -r '"p95 latency (Docker, over HTTP, answered): \(.client_ms_by_status.answered.p95) ms"' \
    eval/bench_groq_vector.json
fi
echo
echo "Details: docs/EVAL_REPORT.md"
