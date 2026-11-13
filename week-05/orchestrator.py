"""Orchestrator: Claude decides which specialists one supplier email needs. Nothing is sent.

Usage (from the top of the repository):
    .venv\\Scripts\\python.exe week-05\\orchestrator.py --email week-05\\brookfield-bikes\\supplier-emails\\01-northgate-controller-delay.eml
    Add --save run.txt to also write everything printed to that UTF-8 text file.

Claude gets the system prompt in brookfield-bikes/prompts/orchestrator.txt and four tools:
    run_supply_analyst     Runs the analyst on the email and returns its brief.
    draft_supplier_reply   Drafts, reviews and saves a reply to the supplier.
    draft_customer_notice  Drafts, reviews and saves a notice for one order at risk.
    finish                 Ends the job with a summary and whether a person must decide.

Rules enforced here in code, not by Claude:
    - The run stops after 10 tool calls, or on any reply that isn't a tool request.
    - Every draft goes through the review step in specialists.py.
    - The analyst runs once. The drafting tools need its brief first.
    - A customer notice is only for an order the brief lists as at risk, once per order.
    - One supplier reply per run.
"""

import json

import pipeline
import specialists

ORCHESTRATOR = "orchestrator"
MAX_TOOL_CALLS = 10

NO_INPUT = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

TOOLS = [
    {
        "name": "run_supply_analyst",
        "description": (
            "Run the supply analyst on the supplier email. It checks stock, scheduled shipments and "
            "customer orders, and returns a brief as JSON: part_id, what_changed, orders_at_risk, "
            "options, decisions_for_a_person, supplier_questions. Runs at most once."
        ),
        "strict": True,
        "input_schema": NO_INPUT,
    },
    {
        "name": "draft_supplier_reply",
        "description": (
            "Have the supplier writer draft a reply to the supplier from the email and the analyst's "
            "brief (customer names are removed first). The draft is reviewed, revised if needed, and "
            "saved for a person. Nothing is sent. Needs the analyst's brief. Runs at most once."
        ),
        "strict": True,
        "input_schema": NO_INPUT,
    },
    {
        "name": "draft_customer_notice",
        "description": (
            "Have the customer writer draft a notice to the customer of one order at risk, given only "
            "that order's facts. The draft is reviewed, revised if needed, and saved for a person. "
            "Nothing is sent. The order must be in the brief's orders_at_risk. One notice per order."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "An order ID from orders_at_risk, such as O-1043."}},
            "required": ["order_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "finish",
        "description": "End the job. Call this once, last.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "A short summary for Brookfield's purchasing manager."},
                "decision_needed": {"type": "boolean", "description": "True if a person must make a decision."},
            },
            "required": ["summary", "decision_needed"],
            "additionalProperties": False,
        },
    },
]


class Job:
    """What the orchestrator has done so far for one email."""

    def __init__(self, email_text):
        self.email_text = email_text
        self.brief = None
        self.drafts = []
        self.supplier_reply_done = False
        self.notices_done = set()
        self.finished = None  # the finish tool's input, once called

    def draft_result(self, draft):
        if draft.path is None:
            return {"no_reply_needed": True, "saved": False}
        return {
            "saved_as": draft.path.name,
            "kind": draft.kind,
            "to": draft.to,
            "reviewer": draft.verdict,
            "revisions": draft.revisions,
            "needs_person": draft.needs_person,
            "reasons": draft.reasons,
        }

    def run_tool(self, name, tool_input):
        """Return (result text, is_error)."""
        if name == "run_supply_analyst":
            if self.brief is not None:
                return "The analyst already ran. Its brief is in an earlier result.", True
            self.brief = specialists.run_analyst(self.email_text)
            return json.dumps(self.brief, ensure_ascii=False), False

        if name == "draft_supplier_reply":
            if self.brief is None:
                return "Run run_supply_analyst first; the supplier writer needs its brief.", True
            if self.supplier_reply_done:
                return "A supplier reply was already drafted in this run.", True
            self.supplier_reply_done = True
            draft = specialists.draft_supplier_reply(self.email_text, self.brief)
            self.drafts.append(draft)
            return json.dumps(self.draft_result(draft), ensure_ascii=False), False

        if name == "draft_customer_notice":
            if self.brief is None:
                return "Run run_supply_analyst first; notices are only for orders it finds at risk.", True
            order_id = str(tool_input.get("order_id", "")).strip().upper()
            if order_id in self.notices_done:
                return f"A notice for {order_id} was already drafted in this run.", True
            order = specialists.customer_order(order_id, self.brief)  # refuses orders not at risk
            self.notices_done.add(order_id)
            draft = specialists.draft_customer_notice(order)
            self.drafts.append(draft)
            return json.dumps(self.draft_result(draft), ensure_ascii=False), False

        if name == "finish":
            self.finished = tool_input
            return "Finished.", False

        return f"Unknown tool: {name}", True


def orchestrate(job):
    """Run the orchestrator's tool loop. Returns a reason if it stopped without finish, else None."""
    system = specialists.read_prompt("orchestrator")
    messages = [{
        "role": "user",
        "content": f"Coordinate Brookfield's response to this supplier email.\n\n<email>\n{job.email_text}\n</email>",
    }]
    calls = 0

    while True:
        response = specialists.ask_claude(ORCHESTRATOR, system=system, tools=TOOLS, messages=messages)
        said = specialists.text_of(response)
        if said:
            print(f"\nOrchestrator said: {said}")
        if response.stop_reason != "tool_use":
            return f"The orchestrator replied without a tool request (stop reason: {response.stop_reason})."

        # Sent back unchanged, including any thinking blocks.
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in (b for b in response.content if b.type == "tool_use"):
            if calls >= MAX_TOOL_CALLS or job.finished is not None:
                why = f"the limit of {MAX_TOOL_CALLS} tool calls was reached" if job.finished is None else "finish was already called"
                print(f"\nNot run, because {why}: {block.name}")
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": f"Not run: {why}.", "is_error": True})
                continue
            calls += 1
            args = f" {json.dumps(block.input, ensure_ascii=False)}" if block.input else ""
            print(f"\n=== Orchestrator choice {calls}: {block.name}{args} ===")
            try:
                text, is_error = job.run_tool(block.name, block.input)
            except specialists.SpecialistError as e:
                text, is_error = f"The specialist could not finish: {e}", True
            if is_error:
                print(f"Refused or failed: {text}")
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": text, "is_error": is_error})

        if job.finished is not None:
            return None
        if calls >= MAX_TOOL_CALLS:
            return f"Reached the limit of {MAX_TOOL_CALLS} tool calls without finish."
        messages.append({"role": "user", "content": results})


def run(email_path):
    try:
        email_text = specialists.load_email(email_path)
    except OSError as e:
        print(f"Could not read the email {email_path} ({type(e).__name__}).")
        return 1
    print(f"Supplier email: {email_path}")

    job = Job(email_text)
    try:
        stopped = orchestrate(job)
    except specialists.SpecialistError as e:  # the orchestrator's own call failed
        stopped = str(e)

    if job.finished is not None:
        print("\n=== Orchestrator's summary ===")
        print(job.finished.get("summary", "").strip())
        print(f"Decision needed (orchestrator's view): {'YES' if job.finished.get('decision_needed') else 'no'}")
    pipeline.print_summary(job.drafts, job.brief, stopped)
    return 1 if stopped else 0


def main():
    pipeline.cli(run, "Let Claude coordinate the specialists for one supplier email. Nothing is sent.")


if __name__ == "__main__":
    main()
