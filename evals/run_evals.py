"""Score shellmate's first step on evals/cases.jsonl. Nothing is ever executed.

Each case is a request plus what a good first step looks like:

    expect_action  the action(s) that count, e.g. "run" or "read_file|run"
    expect         a regex the proposed command (or tool path) must match; "" accepts anything
    expect_risk    the risk level the rules must give that command
    reference      a known-good command (or path), used by --check

Two modes:

    python evals/run_evals.py --check
        No model. Checks the eval set itself: every reference matches its regex and gets the
        expected risk from the rules. This is what CI runs.

    python evals/run_evals.py --model qwen2.5:3b,qwen2.5:7b [--limit 10] [--out results.json]
        Asks each model for the first step of every case and scores it: right action, command
        matches, rules label it as expected. Prints a Markdown table for the README.
        This calls your local model once per case: run it when the machine can take the load.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shellmate.agent import TOOL_ACTIONS, build_messages, clean_command  # noqa: E402
from shellmate.risk import classify  # noqa: E402

CASES = Path(__file__).with_name("cases.jsonl")
REQUIRED = {"id", "request", "expect_action", "expect", "expect_risk", "reference"}


def load_cases(path: Path = CASES) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _actions(case: dict) -> list[str]:
    return case["expect_action"].split("|")


def check_cases(cases: list[dict], *, cwd: str = str(ROOT)) -> list[str]:
    """Problems with the eval set itself (empty list = consistent). Needs no model."""
    problems, seen = [], set()
    for case in cases:
        cid = case.get("id", "?")
        missing = REQUIRED - case.keys()
        if missing:
            problems.append(f"{cid}: missing {sorted(missing)}")
            continue
        if cid in seen:
            problems.append(f"{cid}: duplicate id")
        seen.add(cid)
        try:
            pattern = re.compile(case["expect"])
        except re.error as exc:
            problems.append(f"{cid}: bad regex: {exc}")
            continue
        ref = case["reference"]
        if not ref:
            continue
        if not pattern.search(ref):
            problems.append(f"{cid}: reference {ref!r} does not match {case['expect']!r}")
        if _actions(case)[0] == "run":
            risk = classify(ref, cwd=cwd).risk.value
            if risk != case["expect_risk"]:
                problems.append(f"{cid}: rules label {ref!r} {risk}, case expects {case['expect_risk']}")
    return problems


def score_case(decider, case: dict, *, cwd: str = str(ROOT)) -> dict:
    """Ask for the first step of ``case`` and score it. Never runs the command."""
    started = time.perf_counter()
    try:
        decision = decider.invoke(build_messages(case["request"], [], cwd=cwd))
    except Exception as exc:  # unreachable model, bad JSON ...
        return {"id": case["id"], "error": str(exc), "passed": False,
                "seconds": round(time.perf_counter() - started, 2)}
    seconds = time.perf_counter() - started

    action = decision.action
    if action == "run":
        target = clean_command(decision.command)
        risk = classify(target, cwd=cwd).risk.value
    elif action in TOOL_ACTIONS:
        target, risk = decision.path.strip(), "safe"
    else:
        target, risk = "", "safe"

    action_ok = action in _actions(case)
    match_ok = action_ok and (not case["expect"] or re.search(case["expect"], target) is not None)
    risk_ok = risk == case["expect_risk"]
    return {
        "id": case["id"],
        "action": action,
        "target": target,
        "risk": risk,
        "action_ok": action_ok,
        "match_ok": match_ok,
        "risk_ok": risk_ok,
        "passed": action_ok and match_ok and risk_ok,
        "seconds": round(seconds, 2),
    }


def summarize(results: list[dict]) -> dict:
    n = len(results) or 1

    def share(key: str) -> float:
        return sum(1 for r in results if r.get(key)) / n

    return {
        "cases": len(results),
        "passed": share("passed"),
        "action": share("action_ok"),
        "command": share("match_ok"),
        "risk": share("risk_ok"),
        "errors": sum(1 for r in results if "error" in r),
        "mean_seconds": round(sum(r.get("seconds", 0) for r in results) / n, 2),
    }


def markdown_table(summaries: dict[str, dict]) -> str:
    def pct(x: float) -> str:
        return f"{x * 100:.0f}%"

    rows = [
        "| Model | Pass | Right action | Command matches | Risk label | Errors | Mean time |",
        "|---|---|---|---|---|---|---|",
    ]
    for model, s in summaries.items():
        rows.append(
            f"| {model} | {pct(s['passed'])} | {pct(s['action'])} | {pct(s['command'])} | "
            f"{pct(s['risk'])} | {s['errors']} | {s['mean_seconds']}s |"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None, *, decider_factory=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Check the eval set only; no model.")
    parser.add_argument("--model", default="", help="Comma-separated Ollama models to score.")
    parser.add_argument("--limit", type=int, default=0, help="Only the first N cases.")
    parser.add_argument("--case", action="append", default=[], help="Only these case ids (repeatable).")
    parser.add_argument("--out", type=Path, help="Write every result as JSON here.")
    args = parser.parse_args(argv)

    cases = load_cases()
    problems = check_cases(cases)
    for p in problems:
        print("PROBLEM", p)
    print(f"{len(cases)} cases, {len(problems)} problems")
    if args.check or not args.model:
        if not args.check:
            print("pass --model qwen2.5:3b to score a model (this calls it once per case)")
        return 1 if problems else 0

    if args.case:
        cases = [c for c in cases if c["id"] in set(args.case)]
    if args.limit:
        cases = cases[: args.limit]

    if decider_factory is None:
        from shellmate.agent import build_decider as decider_factory

    all_results, summaries = {}, {}
    for model in [m.strip() for m in args.model.split(",") if m.strip()]:
        decider = decider_factory(model)
        results = []
        for case in cases:
            result = score_case(decider, case)
            results.append(result)
            mark = "ok  " if result["passed"] else "FAIL"
            detail = result.get("target", result.get("error", ""))
            print(f"{mark} {model:14} {case['id']:22} {result.get('action', '-'):10} {detail}")
        all_results[model] = results
        summaries[model] = summarize(results)

    print()
    print(markdown_table(summaries))
    if args.out:
        args.out.write_text(json.dumps({"summaries": summaries, "results": all_results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
