"""Score shellmate on evals/cases.jsonl (milestone 6).

Each case: a request, a regex the proposed command must match, and the expected risk label.
The scorer will report command-match rate and risk accuracy per model, and never execute anything:
it only asks the agent to *propose* commands.

Usage (once implemented):  python evals/run_evals.py --model qwen2.5:3b
"""

from __future__ import annotations

import json
from pathlib import Path

CASES = Path(__file__).with_name("cases.jsonl")


def load_cases() -> list[dict]:
    return [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]


def main() -> None:
    cases = load_cases()
    print(f"{len(cases)} eval cases loaded; scoring lands in milestone 6.")


if __name__ == "__main__":
    main()
