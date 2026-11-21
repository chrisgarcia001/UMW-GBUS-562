"""Run every test case in alder-creek/test-cases.csv through the assistant and check each reply in code.

    python evaluate.py
    python evaluate.py --save results.txt
"""

import argparse
import csv
import sys
from pathlib import Path

import anthropic

from assistant import HERE, AssistantError, assist, load_api_key, unexpected_amounts

CASES_FILE = HERE / "alder-creek" / "test-cases.csv"


def phrases(cell):
    """Split a "a|b|c" cell into its phrases, ignoring blanks."""
    return [p.strip() for p in (cell or "").split("|") if p.strip()]


def check(case, reply):
    """Return the reasons this reply fails the case (an empty list means it passes)."""
    reasons = []
    text = reply["reply"]
    lowered = text.lower()

    expected = case["expected_priority"].strip()
    if reply["priority"] != expected:
        reasons.append(f"priority is {reply['priority']}, expected {expected}")

    must_say = phrases(case["must_say"])
    if must_say and not any(p.lower() in lowered for p in must_say):
        reasons.append("reply has none of: " + ", ".join(f'"{p}"' for p in must_say))

    said = [p for p in phrases(case["must_not_say"]) if p.lower() in lowered]
    if said:
        reasons.append("reply says: " + ", ".join(f'"{p}"' for p in said))

    if case["needs_person"].strip().lower() == "yes" and reply["needs_person"] is not True:
        reasons.append("needs_person is false, expected true")

    bad = unexpected_amounts(text)
    if bad:
        reasons.append("unexpected dollar amount: " + ", ".join(bad))

    return reasons


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Alder Creek assistant against the test cases.")
    parser.add_argument("--save", metavar="FILE", help="also write the output to this file")
    args = parser.parse_args()

    try:
        with CASES_FILE.open(encoding="utf-8-sig", newline="") as f:
            cases = list(csv.DictReader(f))
        client = anthropic.Anthropic(api_key=load_api_key())
    except (OSError, AssistantError) as e:
        print(e, file=sys.stderr)
        return 1

    # Open the save file before any calls to Claude, so a bad path fails before money is spent.
    save_file = None
    if args.save:
        try:
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            save_file = open(args.save, "w", encoding="utf-8")
        except OSError as e:
            print(f"Could not save to {args.save}: {e}", file=sys.stderr)
            return 1

    try:
        return run_cases(cases, client, save_file, args.save)
    finally:
        if save_file:
            save_file.close()


def run_cases(cases, client, save_file, save_name):
    def out(line=""):
        print(line, flush=True)
        if save_file:
            save_file.write(line + "\n")
            save_file.flush()  # Each line is kept even if the run is stopped part-way.

    passed = 0
    for case in cases:
        label = f"{case['id']} ({case['type']})"
        try:
            reply = assist(case["request"], client=client)
        except AssistantError as e:
            out(f"{label}: FAIL - no reply: {e}")
            continue
        reasons = check(case, reply)
        if reasons:
            out(f"{label}: FAIL - " + "; ".join(reasons))
            # Show what the guardrails did with this reply.
            out(f"    needs_person: {str(reply['needs_person']).lower()}")
            out(f"    held_because: {reply.get('held_because', '(not held)')}")
        else:
            out(f"{label}: PASS")
            passed += 1

    out()
    out(f"Passed {passed} of {len(cases)}.")

    if save_file:
        print(f"Saved to {save_name}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
