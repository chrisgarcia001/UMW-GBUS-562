"""Fixed-order pipeline: one supplier email in, reviewed drafts out. Nothing is sent.

Usage (from the top of the repository):
    .venv\\Scripts\\python.exe week-05\\pipeline.py --email week-05\\brookfield-bikes\\supplier-emails\\01-northgate-controller-delay.eml
    Add --save run.txt to also write everything printed to that UTF-8 text file.

Steps, always in this order:
    1. The supply analyst writes a brief.
    2. A customer notice for each order at risk (skipped if there are none).
    3. A reply to the supplier.
Every draft goes through the review step in specialists.py and is saved in week-05/drafts/.
"""

import argparse
import json
import sys
from pathlib import Path

import specialists


class Tee:
    """Writes everything printed to the screen and to a file."""

    def __init__(self, screen, file):
        self.screen = screen
        self.file = file

    def write(self, text):
        self.file.write(text)
        return self.screen.write(text)

    def flush(self):
        self.file.flush()
        self.screen.flush()


def step(number, title):
    print(f"\n=== Step {number}: {title} ===")


def print_summary(drafts, brief, stopped):
    saved = [d for d in drafts if d.path]
    marked = [d for d in saved if d.needs_person]

    print("\n=== Summary ===")
    if stopped:
        print(f"Stopped early: {stopped}")
    print(f"Drafts saved ({len(saved)}):")
    for d in saved:
        print(f"  {d.kind} to {d.to}: {d.path}")
    if any(d.kind == "supplier reply" and not d.path for d in drafts):
        print("  (No supplier reply was needed.)")

    print(f"Drafts marked for a person ({len(marked)}):")
    for d in marked:
        print(f"  {d.path.name}: {'; '.join(d.reasons)}")

    decisions = brief.get("decisions_for_a_person", []) if brief else []
    print(f"Decisions for a person ({len(decisions)}):")
    for item in decisions:
        print(f"  - {item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)}")

    specialists.print_token_totals()
    print("\nNothing was sent.")


def run(email_path):
    try:
        email_text = specialists.load_email(email_path)
    except OSError as e:
        print(f"Could not read the email {email_path} ({type(e).__name__}).")
        return 1
    print(f"Supplier email: {email_path}")

    drafts, brief, stopped = [], None, None
    try:
        step(1, "supply analyst")
        brief = specialists.run_analyst(email_text)

        at_risk = brief["orders_at_risk"]
        if at_risk:
            step(2, f"customer notices ({len(at_risk)} orders at risk)")
            for risk in at_risk:
                print(f"\n--- Customer notice for {risk['order_id']} ---")
                order = specialists.customer_order(risk["order_id"], brief)
                drafts.append(specialists.draft_customer_notice(order))
        else:
            step(2, "customer notices: skipped, no orders at risk")

        step(3, "reply to the supplier")
        drafts.append(specialists.draft_supplier_reply(email_text, brief))
    except specialists.SpecialistError as e:
        stopped = str(e)

    print_summary(drafts, brief, stopped)
    return 1 if stopped else 0


def cli(run, description):
    """Parse --email and --save, then call run(email_path). Shared with orchestrator.py."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--email", type=Path, required=True, help="path of one supplier email (.eml)")
    parser.add_argument("--save", type=Path, metavar="FILE",
                        help="also write everything printed to this UTF-8 text file, replacing it if it exists")
    args = parser.parse_args()

    if not args.save:
        sys.exit(run(args.email))

    # Opened first, so a bad path stops the program before any tokens are spent.
    try:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        save_file = args.save.open("w", encoding="utf-8")
    except OSError as e:
        sys.exit(f"Could not open {args.save} for writing ({type(e).__name__}).")
    screen = sys.stdout
    sys.stdout = Tee(screen, save_file)
    try:
        status = run(args.email)
    finally:
        sys.stdout = screen
        save_file.close()
    sys.exit(status)


def main():
    cli(run, "Prepare reviewed drafts for one supplier email. Nothing is sent.")


if __name__ == "__main__":
    main()
