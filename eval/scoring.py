"""Pure scoring rules for the evaluation set (D48, D49). No I/O, no LLM: unit-tested.

Matching: both the answer and each expected phrase are normalised (lowercase, commas stripped,
₹ / Rs / Rs. / INR unified, "p.m." -> "pm"), then matched on WORD BOUNDARIES, so "hr" does not
match "three" and "3" does not match "30".

Pass rule for one case:
  status matches
  AND (if answered) the cited sections include the expected one(s), from the expected version
  AND every must_include group has at least one alternative present
  AND no must_not_include phrase is present
  AND no retrieved or cited chunk belongs to another company (leak)
  AND llm_calls <= 2 and citations <= 3 (the brief's budgets; hard, on every case).
The gate that fired (tau vs LLM) is reported, not pass-critical (D49).
"""

import re
from dataclasses import dataclass, field

CHUNK_ID_RE = re.compile(r"^(?P<co>[a-z0-9]+)_v(?P<v>\d+)_s(?P<sid>[0-9.]+)_p(?P<part>\d+)$")
MAX_LLM_CALLS, MAX_CITATIONS = 2, 3  # from the brief (D49)
_CURRENCY_RE = re.compile(r"(?:₹|(?<![a-z0-9])rs\.?|(?<![a-z0-9])inr(?![a-z0-9]))\s*")


def normalise(text: str) -> str:
    t = text.lower()
    t = t.replace("p.m.", "pm").replace("a.m.", "am")
    t = t.replace(",", "")
    t = _CURRENCY_RE.sub("₹", t)
    return re.sub(r"\s+", " ", t).strip()


def contains(text: str, phrase: str) -> bool:
    """Word-boundary match of a normalised phrase inside normalised text."""
    p = normalise(phrase)
    return re.search(rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])", normalise(text)) is not None


def parse_chunk_id(chunk_id: str) -> tuple[str, int, str] | None:
    m = CHUNK_ID_RE.match(chunk_id)
    return (m["co"], int(m["v"]), m["sid"]) if m else None


@dataclass
class Score:
    passed: bool
    failures: list[str] = field(default_factory=list)
    sections_ok: bool | None = None  # None when not applicable (expected not_covered)
    facts_ok: bool | None = None
    hit_at_3: bool | None = None
    leak: bool = False
    gate_as_expected: bool | None = None


def score_case(case: dict, actual: dict) -> Score:
    exp, company = case["expected"], case["input"]["company"]
    failures: list[str] = []

    # Isolation: anything retrieved or cited from another company is a leak (must never happen).
    seen = list(actual.get("retrieved_ids", [])) + list(actual.get("citations", []))
    leak = any((p := parse_chunk_id(cid)) is None or p[0] != company for cid in seen)
    if leak:
        failures.append("LEAK: chunk from another company")

    if actual.get("llm_calls", 0) > MAX_LLM_CALLS:
        failures.append(f"BUDGET: {actual['llm_calls']} LLM calls > {MAX_LLM_CALLS}")
    if len(actual.get("citations", [])) > MAX_CITATIONS:
        failures.append(f"BUDGET: {len(actual['citations'])} citations > {MAX_CITATIONS}")

    if actual["status"] != exp["status"]:
        failures.append(f"status {actual['status']} != expected {exp['status']}")

    wanted = {(company, exp["version"], sid) for sid in exp["sections"]}
    hit_at_3 = sections_ok = facts_ok = None
    if exp["status"] == "answered":
        top3 = {parse_chunk_id(c) for c in actual.get("retrieved_ids", [])[:3]}
        hit_at_3 = bool(wanted & {(p[0], p[1], p[2]) for p in top3 if p})
        if actual["status"] == "answered":
            cited = {p for c in actual.get("citations", []) if (p := parse_chunk_id(c))}
            cited = {(p[0], p[1], p[2]) for p in cited}
            sections_ok = (
                wanted <= cited if exp.get("require_all_sections") else bool(wanted & cited)
            )
            if not sections_ok:
                need = "all of" if exp.get("require_all_sections") else "one of"
                failures.append(f"citations miss {need} §{', §'.join(exp['sections'])}")

    answer = actual.get("answer", "")
    missing = [g for g in exp["must_include"] if not any(contains(answer, alt) for alt in g)]
    present = [p for p in exp["must_not_include"] if contains(answer, p)]
    for g in missing:
        failures.append(f"missing one of {g}")
    for p in present:
        failures.append(f"contains forbidden '{p}'")
    if exp["status"] == "answered" and actual["status"] == "answered":
        facts_ok = not missing and not present

    gate_as_expected = None
    if "gate" in exp:
        gate_as_expected = actual.get("gate") == exp["gate"]

    return Score(
        passed=not failures,
        failures=failures,
        sections_ok=sections_ok,
        facts_ok=facts_ok,
        hit_at_3=hit_at_3,
        leak=leak,
        gate_as_expected=gate_as_expected,
    )


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile (no interpolation), e.g. pct=95 -> p95."""
    if not values:
        return None
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, -(-len(ordered) * pct // 100) - 1))
    return ordered[int(k)]
