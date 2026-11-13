"""The four Brookfield Bicycle Company specialists that the coordinators share.

Specialists (each reads its system prompt from brookfield-bikes/prompts/ on every run):
    run_analyst(email)                  An agent with the tools of supply_mcp_server.py. Stops
                                        after 10 tool calls and returns its brief as a dict.
    write_supplier_reply(email, brief)  One model call. Customer names are removed first.
    write_customer_notice(order)        One model call, given only that one order's facts.
    review(draft, facts, recipient)     One model call. Returns ("APPROVE" or "REVISE", reasons).

Review steps (what a coordinator normally calls):
    draft_supplier_reply(email, brief)  Write, review, revise at most twice, check, save.
    draft_customer_notice(order)        The same for a notice to one customer. Build the
                                        order with customer_order(order_id, brief).

After the reviewer, code checks that no customer name is in a supplier draft and that no
supplier name (or other customer's name) is in a customer draft. Drafts are saved in
week-05/drafts/ and marked for a person if the reviewer never approved, the name check
failed, or (for a customer notice) the order has a late penalty or the supplier offered
options, since the delay then depends on Brookfield's choice. Nothing is ever sent.

Every handoff between the coordinator and a specialist is printed, and so are the tokens
each call used. print_token_totals() prints the totals per specialist.

The API key is read from API-Key.txt in the folder above the repository. It is only ever
passed to the Anthropic client.

Example (from the top of the repository):
    import specialists
    email = specialists.load_email(Path("week-05/brookfield-bikes/supplier-emails/01-northgate-controller-delay.eml"))
    brief = specialists.run_analyst(email)
    specialists.draft_supplier_reply(email, brief)
    for risk in brief["orders_at_risk"]:
        specialists.draft_customer_notice(specialists.customer_order(risk["order_id"], brief))
    specialists.print_token_totals()
"""

import asyncio
import email as email_lib
import email.policy
import json
import re
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import anthropic
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client

import supply_tools

HERE = Path(__file__).resolve().parent
SERVER = HERE / "supply_mcp_server.py"
PROMPTS_DIR = HERE / "brookfield-bikes" / "prompts"
DRAFTS_DIR = HERE / "drafts"
# API-Key.txt lives in the folder that contains the repository (never inside it).
API_KEY_FILE = HERE.parent.parent / "API-Key.txt"

MODEL = "claude-sonnet-5"
MAX_TOKENS = 16000
MAX_TOOL_CALLS = 10
MAX_REVISIONS = 2
NO_REPLY = "NO REPLY NEEDED"

COORDINATOR = "coordinator"
ANALYST = "supply analyst"
SUPPLIER_WRITER = "supplier writer"
CUSTOMER_WRITER = "customer writer"
REVIEWER = "reviewer"

BRIEF_LISTS = ("orders_at_risk", "options", "decisions_for_a_person", "supplier_questions")

# The only order facts the customer writer receives. late_penalty and contact_email stay out.
NOTICE_FIELDS = ("order_id", "customer", "model", "bikes", "planned_ship_day", "expected_ship_day", "days_late")


class SpecialistError(Exception):
    """A specialist could not finish. The message explains why and never holds the key."""


# ---------------------------------------------------------------------------
# Printing handoffs and tokens
# ---------------------------------------------------------------------------

TOKENS = {}


def handoff(sender, receiver, what, content):
    if not isinstance(content, str):
        content = json.dumps(content, indent=2, ensure_ascii=False)
    print(f"\n----- HANDOFF: {sender} -> {receiver} ({what}) -----")
    print(content.strip())
    print("-" * 60)


def record_tokens(specialist, usage):
    totals = TOKENS.setdefault(specialist, {"calls": 0, "input": 0, "output": 0})
    totals["calls"] += 1
    totals["input"] += usage.input_tokens
    totals["output"] += usage.output_tokens
    print(f"[tokens] {specialist}, call {totals['calls']}: {usage.input_tokens} input, {usage.output_tokens} output")


def print_token_totals():
    print("\n=== Tokens used by each specialist ===")
    all_in = all_out = 0
    for specialist, t in TOKENS.items():
        print(f"{specialist}: {t['input']} input, {t['output']} output ({t['calls']} calls)")
        all_in += t["input"]
        all_out += t["output"]
    print(f"All specialists: {all_in} input, {all_out} output")


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------

_client = None


def get_client():
    global _client
    if _client is None:
        try:
            key = API_KEY_FILE.read_text(encoding="utf-8").strip()
        except OSError:
            raise SpecialistError(f"The API key file could not be read: {API_KEY_FILE}") from None
        if not key:
            raise SpecialistError(f"The API key file is empty: {API_KEY_FILE}")
        _client = anthropic.Anthropic(api_key=key)
    return _client


def read_prompt(name):
    path = PROMPTS_DIR / f"{name}.txt"
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        raise SpecialistError(f"The system prompt could not be read: {path}") from None


def ask_claude(specialist, **request):
    """Make one request, record its tokens, and return the response."""
    try:
        response = get_client().messages.create(model=MODEL, max_tokens=MAX_TOKENS, **request)
    except anthropic.AuthenticationError:
        raise SpecialistError(f"Claude rejected the API key in {API_KEY_FILE.name}.") from None
    except anthropic.NotFoundError:
        raise SpecialistError(f"Model {MODEL} was not found.") from None
    except anthropic.RateLimitError:
        raise SpecialistError("Rate limited by the API. Wait a moment and try again.") from None
    except anthropic.APIStatusError as e:
        raise SpecialistError(f"API error {e.status_code}: {e.message}") from None
    except anthropic.APIConnectionError:
        raise SpecialistError("Could not reach the Claude API. Check your internet connection.") from None
    record_tokens(specialist, response.usage)
    if response.stop_reason in ("refusal", "max_tokens"):
        raise SpecialistError(f"The {specialist} stopped early (stop reason: {response.stop_reason}).")
    return response


def text_of(response):
    return "".join(b.text for b in response.content if b.type == "text").strip()


# ---------------------------------------------------------------------------
# Data from the CSV files
# ---------------------------------------------------------------------------


def load_email(path):
    """Return an .eml file as plain text: From, To and Subject lines, then the body."""
    msg = email_lib.message_from_bytes(Path(path).read_bytes(), policy=email.policy.default)
    body_part = msg.get_body(preferencelist=("plain",))
    body = body_part.get_content().strip() if body_part else ""
    headers = "\n".join(f"{h}: {msg[h]}" for h in ("From", "To", "Subject") if msg[h])
    return f"{headers}\n\n{body}"


def header_line(email_text, name):
    match = re.search(rf"^{name}:\s*(.*)$", email_text, re.MULTILINE | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def customer_names():
    """Every customer name and contact address in orders.csv."""
    rows = supply_tools.read_csv("orders.csv")
    return {r[k].strip() for r in rows for k in ("customer", "contact_email") if r.get(k, "").strip()}


def supplier_names():
    """Every supplier name, contact name and contact address in suppliers.csv and parts.csv."""
    names = {r["supplier"].strip() for r in supply_tools.read_csv("parts.csv")}
    for r in supply_tools.read_csv("suppliers.csv"):
        names |= {r[k].strip() for k in ("supplier", "contact_name", "contact_email") if r.get(k, "").strip()}
    return names


def name_pattern(names):
    """A case-insensitive pattern matching any of the names as whole words."""
    parts = [r"\s+".join(re.escape(w) for w in n.split()) for n in sorted(names, key=len, reverse=True)]
    return re.compile(r"(?<![\w@.])(?:" + "|".join(parts) + r")(?![\w@])", re.IGNORECASE)


def names_in(text, names):
    return sorted({m.group(0) for m in name_pattern(names).finditer(text)}) if names else []


def without_customer_names(value, pattern=None):
    """Return a copy of a brief (or text) with every customer name removed.

    "customer" fields are dropped, and any customer name or address left in the text
    becomes "a customer".
    """
    pattern = pattern or name_pattern(customer_names())
    if isinstance(value, dict):
        return {k: without_customer_names(v, pattern) for k, v in value.items()
                if k not in ("customer", "contact_email")}
    if isinstance(value, list):
        return [without_customer_names(v, pattern) for v in value]
    if isinstance(value, str):
        return pattern.sub("a customer", value)
    return value


def customer_order(order_id, brief):
    """Return the facts for one order at risk in the brief, taken from orders.csv and the brief.

    Raises SpecialistError if the brief doesn't list the order as at risk, so a notice can
    only go to a customer whose order the analyst found at risk.
    """
    wanted = str(order_id).strip().upper()
    risk = next((r for r in brief["orders_at_risk"] if str(r.get("order_id", "")).strip().upper() == wanted), None)
    if risk is None:
        raise SpecialistError(f"{order_id} is not an order at risk in the analyst's brief.")
    row = next((r for r in supply_tools.read_csv("orders.csv") if r["order_id"].strip().upper() == wanted), None)
    if row is None:
        raise SpecialistError(f"{order_id} is not in orders.csv.")

    planned = int(row["ship_in_days"])
    ready = risk.get("ready_day")
    ready = ready if isinstance(ready, int) and not isinstance(ready, bool) else None
    return {
        "order_id": row["order_id"],
        "customer": row["customer"],
        "contact_email": row["contact_email"],
        "model": row["model"],
        "bikes": int(row["qty"]),
        "planned_ship_day": planned,
        "expected_ship_day": ready if ready is not None else "not yet known",
        "days_late": ready - planned if ready is not None else "not yet known",
        "late_penalty": row["late_penalty"].strip().lower() == "yes",
        # If the supplier offers options, the delay may not happen, depending on Brookfield's choice.
        "depends_on_option": bool(brief.get("options")),
    }


# ---------------------------------------------------------------------------
# Specialist 1: the supply analyst (an agent with the MCP server's tools)
# ---------------------------------------------------------------------------


def parse_brief(text):
    """Return the first JSON object in the text that has every brief field."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            obj = None
        if (isinstance(obj, dict) and isinstance(obj.get("part_id"), str)
                and isinstance(obj.get("what_changed"), str)
                and all(isinstance(obj.get(k), list) for k in BRIEF_LISTS)):
            if all(isinstance(r, dict) and r.get("order_id") for r in obj["orders_at_risk"]):
                return obj
        start = text.find("{", start + 1)
    raise SpecialistError("The analyst did not return a brief as JSON with all the required fields.")


async def call_mcp_tool(session, name, tool_input):
    """Return (result text, is_error)."""
    result = await session.call_tool(name, tool_input)
    text = "\n".join(block.text for block in result.content if block.type == "text")
    return text, bool(result.is_error)


async def analyst_loop(session, email_text):
    """Run the analyst's tool loop and return its final text."""
    listed = await session.list_tools()
    tool_names = {t.name for t in listed.tools}
    tools = [{"name": t.name, "description": t.description or "", "input_schema": t.input_schema} for t in listed.tools]
    system = read_prompt("analyst")
    messages = [{"role": "user", "content": f"Here is the supplier email.\n\n<email>\n{email_text}\n</email>"}]
    calls = 0

    while True:
        out_of_calls = calls >= MAX_TOOL_CALLS
        extra = {"tool_choice": {"type": "none"}} if out_of_calls else {}
        response = ask_claude(ANALYST, system=system, tools=tools, messages=messages, **extra)
        if response.stop_reason != "tool_use":
            return text_of(response)
        if out_of_calls:
            raise SpecialistError("The analyst asked for a tool after its tool limit.")

        # Sent back unchanged, including any thinking blocks.
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in (b for b in response.content if b.type == "tool_use"):
            if calls >= MAX_TOOL_CALLS:
                print(f"  Not run, over the limit of {MAX_TOOL_CALLS}: {block.name}")
                text, is_error = f"Not run: the limit of {MAX_TOOL_CALLS} tool calls was reached.", True
            else:
                calls += 1
                print(f"  Analyst tool call {calls}: {block.name} {json.dumps(block.input, ensure_ascii=False)}")
                if block.name in tool_names:
                    text, is_error = await call_mcp_tool(session, block.name, block.input)
                else:
                    text, is_error = f"Unknown tool: {block.name}", True
                if is_error:
                    print(f"    Error: {text}")
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": text, "is_error": is_error})

        if calls >= MAX_TOOL_CALLS:
            print(f"  The analyst reached the limit of {MAX_TOOL_CALLS} tool calls.")
            results.append({"type": "text", "text": (
                f"You have used all {MAX_TOOL_CALLS} tool calls. Write your brief now, as JSON, "
                "from the tool results you have."
            )})
        messages.append({"role": "user", "content": results})


async def analyst_with_server(email_text, server_log):
    """Start the MCP server, run the analyst, and return ("ok", text) or ("error", message)."""
    server = StdioServerParameters(command=sys.executable, args=[str(SERVER)], cwd=str(HERE))
    async with Client(stdio_client(server, errlog=server_log)) as session:
        try:
            return "ok", await analyst_loop(session, email_text)
        except SpecialistError as e:
            return "error", str(e)


def run_analyst(email_text):
    """Return the analyst's brief for one supplier email, as a dict."""
    handoff(COORDINATOR, ANALYST, "supplier email", email_text)
    # The server's log goes to a temporary file, so it can be printed here.
    with tempfile.TemporaryFile() as server_log:
        try:
            status, text = asyncio.run(analyst_with_server(email_text, server_log))
        except (Exception, BaseExceptionGroup) as e:  # the MCP client reports a failed server as a group
            if isinstance(e, BaseExceptionGroup) and e.subgroup(KeyboardInterrupt):
                raise KeyboardInterrupt from None
            status, text = "error", f"The supply tools server could not start or stopped responding ({type(e).__name__})."
        server_log.seek(0)
        log_text = server_log.read().decode("utf-8", errors="replace").strip()
    if log_text:
        print(f"Supply tools server log:\n{log_text}")
    if status == "error":
        raise SpecialistError(text)

    brief = parse_brief(text)
    handoff(ANALYST, COORDINATOR, "brief", brief)
    return brief


# ---------------------------------------------------------------------------
# Specialists 2 to 4: the writers and the reviewer (one model call each)
# ---------------------------------------------------------------------------


def revision_request(previous_draft, reviewer_notes):
    if previous_draft is None:
        return "Reply with only the text of the email."
    return (
        f"\n\nYour previous draft:\n<previous_draft>\n{previous_draft}\n</previous_draft>\n\n"
        f"A reviewer asked for changes:\n<reviewer_notes>\n{reviewer_notes}\n</reviewer_notes>\n\n"
        "Revise the draft to fix every point the reviewer raised. Reply with only the text of the revised email."
    )


def supplier_facts(email_text, brief):
    """What the supplier writer and its reviewer see: the email and brief with customer names removed."""
    pattern = name_pattern(customer_names())
    clean_email = without_customer_names(email_text, pattern)
    clean_brief = without_customer_names(brief, pattern)
    return (
        f"<email>\n{clean_email}\n</email>\n\n"
        f"<brief>\n{json.dumps(clean_brief, indent=2, ensure_ascii=False)}\n</brief>"
    )


def write_supplier_reply(email_text, brief, previous_draft=None, reviewer_notes=None):
    """Return a draft reply to the supplier. Customer names are removed before the call."""
    facts = supplier_facts(email_text, brief)
    sender = COORDINATOR if previous_draft is None else REVIEWER
    handoff(sender, SUPPLIER_WRITER, "email and brief, customer names removed" if previous_draft is None else "revision request",
            facts if previous_draft is None else reviewer_notes)
    content = f"Draft a reply to this supplier email.\n\n{facts}\n\n" + revision_request(previous_draft, reviewer_notes)
    draft = text_of(ask_claude(SUPPLIER_WRITER, system=read_prompt("supplier-writer"),
                               messages=[{"role": "user", "content": content}]))
    if not draft:
        raise SpecialistError("The supplier writer returned an empty draft.")
    return draft


def notice_facts(order):
    """Only the facts the customer writer may see, as JSON."""
    return json.dumps({k: order[k] for k in NOTICE_FIELDS if k in order}, indent=2, ensure_ascii=False)


def write_customer_notice(order, previous_draft=None, reviewer_notes=None):
    """Return a draft notice to one customer, written from that order's facts only.

    Days are counted from today: planned_ship_day and expected_ship_day are days from today.
    """
    facts = notice_facts(order)
    sender = COORDINATOR if previous_draft is None else REVIEWER
    handoff(sender, CUSTOMER_WRITER, "order facts" if previous_draft is None else "revision request",
            facts if previous_draft is None else reviewer_notes)
    content = (
        "Draft a notice to this customer. Ship days are counted in days from today.\n\n"
        f"<order_facts>\n{facts}\n</order_facts>\n\n" + revision_request(previous_draft, reviewer_notes)
    )
    draft = text_of(ask_claude(CUSTOMER_WRITER, system=read_prompt("customer-writer"),
                               messages=[{"role": "user", "content": content}]))
    if not draft:
        raise SpecialistError("The customer writer returned an empty draft.")
    return draft


def review(draft, facts, recipient):
    """Return ("APPROVE" or "REVISE", the reviewer's full answer). recipient: "supplier" or "customer".

    An answer that starts with neither word counts as REVISE.
    """
    handoff(COORDINATOR, REVIEWER, f"draft to a {recipient}, with its facts",
            f"<draft>\n{draft}\n</draft>\n\n<facts>\n{facts}\n</facts>")
    content = (
        f"This draft goes to a {recipient}.\n\n"
        f"<facts>\n{facts}\n</facts>\n\n<draft>\n{draft}\n</draft>"
    )
    answer = text_of(ask_claude(REVIEWER, system=read_prompt("reviewer"), messages=[{"role": "user", "content": content}]))
    first = next((line for line in answer.splitlines() if line.strip()), "")
    word = re.sub(r"[^A-Za-z]", "", first).upper()
    verdict = "APPROVE" if word == "APPROVE" else "REVISE"
    handoff(REVIEWER, COORDINATOR, verdict, answer)
    return verdict, answer


# ---------------------------------------------------------------------------
# Review steps
# ---------------------------------------------------------------------------


@dataclass
class Draft:
    kind: str                  # "supplier reply" or "customer notice"
    to: str
    subject: str
    text: str
    verdict: str = "not reviewed"
    revisions: int = 0
    reviewer_notes: str = ""
    names_found: list = field(default_factory=list)
    needs_person: bool = False
    reasons: list = field(default_factory=list)
    path: Path = None


def write_and_review(write, facts, recipient):
    """Run a writer, then the reviewer, revising at most MAX_REVISIONS times.

    write(previous_draft, reviewer_notes) returns a draft. Returns (draft, verdict, notes, revisions).
    """
    draft = write(None, None)
    revisions = 0
    while True:
        verdict, notes = review(draft, facts, recipient)
        if verdict == "APPROVE" or revisions == MAX_REVISIONS:
            return draft, verdict, notes, revisions
        revisions += 1
        draft = write(draft, notes)


def finish(result, forbidden, label, late_penalty=False, depends_on_option=False):
    """Run the name check, decide whether a person must look, and save the draft."""
    result.names_found = names_in(result.text, forbidden)
    if result.verdict != "APPROVE":
        result.reasons.append(f"the reviewer did not approve it after {result.revisions} revisions")
    if result.names_found:
        result.reasons.append(f"it names {', '.join(result.names_found)}")
    if late_penalty:
        result.reasons.append("the order has a late penalty")
    if depends_on_option:
        result.reasons.append("the supplier offered options, so this delay depends on which one Brookfield chooses")
    result.needs_person = bool(result.reasons)
    result.path = save_draft(result, label)
    print(f"\nSaved {result.kind}: {result.path}")
    print(f"Needs a person: {'YES, because ' + '; '.join(result.reasons) if result.needs_person else 'no'}")
    return result


def draft_supplier_reply(email_text, brief):
    """Write, review and save a reply to the supplier. Returns a Draft (path None if no reply is needed)."""
    subject = header_line(email_text, "Subject")
    result = Draft(kind="supplier reply", to=header_line(email_text, "From"),
                   subject=f"Re: {subject}" if subject else "Re: your email", text="")

    first = write_supplier_reply(email_text, brief)
    if first.strip().upper().rstrip(".") == NO_REPLY:
        handoff(SUPPLIER_WRITER, COORDINATOR, "no reply needed", first)
        result.text = NO_REPLY
        print("\nNo supplier reply needed, so nothing was reviewed or saved.")
        return result

    # The first draft is already written; later calls are revisions.
    write = lambda prev, notes: first if prev is None else write_supplier_reply(email_text, brief, prev, notes)
    result.text, result.verdict, result.reviewer_notes, result.revisions = write_and_review(
        write, supplier_facts(email_text, brief), "supplier")
    return finish(result, customer_names(), str(brief.get("part_id", "part")))


def draft_customer_notice(order):
    """Write, review and save a notice to one customer. order comes from customer_order()."""
    result = Draft(kind="customer notice", to=f"{order['customer']} <{order.get('contact_email', '')}>",
                   subject=f"Your order {order['order_id']}", text="")
    write = lambda prev, notes: write_customer_notice(order, prev, notes)
    result.text, result.verdict, result.reviewer_notes, result.revisions = write_and_review(
        write, notice_facts(order), "customer")
    # Suppliers, and every customer other than this one.
    forbidden = supplier_names() | (customer_names() - {order["customer"], order.get("contact_email", "")})
    return finish(result, forbidden, order["order_id"], late_penalty=order.get("late_penalty", False),
                  depends_on_option=order.get("depends_on_option", False))


def save_draft(result, label):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    kind = "supplier" if result.kind == "supplier reply" else "customer"
    flag = "-NEEDS-PERSON" if result.needs_person else ""
    path = DRAFTS_DIR / f"{stamp}-{kind}-{label}{flag}.txt"
    reviewer = f"{result.verdict} after {result.revisions} revision{'s' if result.revisions != 1 else ''}"
    text = (
        "Status: DRAFT, not sent\n"
        f"Kind: {result.kind}\n"
        f"To: {result.to}\n"
        f"Subject: {result.subject}\n"
        f"Reviewer: {reviewer}\n"
        f"Name check: {'FAILED, names ' + ', '.join(result.names_found) if result.names_found else 'passed'}\n"
        f"Needs a person: {'YES, because ' + '; '.join(result.reasons) if result.needs_person else 'no'}\n"
        "\n"
        f"{result.text.strip()}\n"
    )
    try:
        DRAFTS_DIR.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as e:
        raise SpecialistError(f"The draft could not be saved to {path} ({type(e).__name__}).") from None
    return path
