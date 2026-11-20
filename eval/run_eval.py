"""Run the fixed evaluation set and write eval/results.jsonl.

Output fields (fixed by the brief): id, input, expected, actual, pass, notes.
"""

import json
from pathlib import Path

CASES = Path("eval/cases.jsonl")
RESULTS = Path("eval/results.jsonl")


def load_cases() -> list[dict]:
    if not CASES.exists():
        return []
    return [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]


def main() -> None:
    cases = load_cases()
    # TODO: run each case through the API, score retrieval + answer, compute metrics.
    print(f"Loaded {len(cases)} cases. Evaluation not implemented yet.")


if __name__ == "__main__":
    main()
