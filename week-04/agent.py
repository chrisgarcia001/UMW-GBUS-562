"""Command-line agent that prepares replies to HR emails, using Claude's tool use.

Usage (from the top of the repository):
    .venv\\Scripts\\python.exe week-04\\agent.py --email week-04\\hartwell-hr\\sample-emails\\01-dana-vacation.eml
        Handles one email file and writes the draft to week-04/drafts/.
    .venv\\Scripts\\python.exe week-04\\agent.py --inbox
        Handles each unread email in the Gmail inbox and creates real Gmail drafts.
    .venv\\Scripts\\python.exe week-04\\agent.py --watch
        Checks for unread email every 30 seconds and SENDS replies that don't need review.
    Add --save output.txt to any of them to also write everything printed to that file.

For each email, the agent starts hr_mcp_server.py with --as-employee and the sender's
address, so the balance tool can return only the sender's own balance. Claude gets that
server's tools plus save_reply. Each run stops after 10 tool calls, or on any reply from
Claude that isn't a tool request.

--email and --inbox never send anything. Only --watch can send, under rules enforced here
in the code, not by Claude, which has no send tool:
    - Only senders on Agent-Allowlist.txt are handled. Each line reads
      real.address@gmail.com = employee@hartwellfoods.example, and the sender is treated as
      that employee. Other email, and automatic email, stays unread and never reaches Claude.
    - A reply is sent only if needs_human_review is false, and only to the sender's address.
      Otherwise it stays a draft.
    - At most 5 replies are sent, then it stops. Ctrl+C also stops it.

The Gmail login comes from Agent-Email.txt, in the folder above the repository, and is only
ever passed to Gmail. It is never printed or saved, so errors that might repeat it are
reported by type, never by their text. In --watch mode all output also passes through a
filter that hides the login and every allowlisted address.
"""

import argparse
import asyncio
import contextlib
import email
import email.message
import email.policy
import email.utils
import imaplib
import json
import re
import smtplib
import ssl
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import anthropic
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client

import load_inbox

HERE = Path(__file__).resolve().parent
SERVER = HERE / "hr_mcp_server.py"
SYSTEM_PROMPT_FILE = HERE / "hartwell-hr" / "system_prompt.txt"
DRAFTS_DIR = HERE / "drafts"
# API-Key.txt lives in the folder that contains the repository (never inside it).
API_KEY_FILE = HERE.parent.parent / "API-Key.txt"
ALLOWLIST_FILE = HERE.parent.parent / "Agent-Allowlist.txt"

MODEL = "claude-sonnet-5"
MAX_TOKENS = 16000
MAX_TOOL_CALLS = 10
RESULT_PREVIEW_CHARS = 300
REVIEW_LABEL = "NEEDS HR REVIEW"

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465
WATCH_SECONDS = 30
MAX_SENT = 5
MAX_CONNECTION_FAILURES = 5
ADDRESS_PATTERN = r"[^@\s]+@[^@\s]+"
HIDDEN = "[address hidden]"

SAVE_REPLY_TOOL = {
    "name": "save_reply",
    "description": (
        "Save your finished reply as a draft for the HR team to review. This only saves it; "
        "nothing is sent. Call this exactly once, when the reply is complete."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "reply_text": {"type": "string", "description": "The full text of the reply to the employee."},
            "needs_human_review": {
                "type": "boolean",
                "description": "True if a person from HR must handle this email, as your instructions describe.",
            },
        },
        "required": ["reply_text", "needs_human_review"],
        "additionalProperties": False,
    },
}


# Claude is told what really happens to its reply, but still has no way to send anything.
WATCH_SAVE_REPLY_TOOL = {
    **SAVE_REPLY_TOOL,
    "description": (
        "Submit your finished reply. Call this exactly once, when the reply is complete. If "
        "needs_human_review is false, this program sends the reply to the employee right away "
        "with no person checking it. If it's true, the reply stays a draft for the HR team."
    ),
}


class SetupError(Exception):
    """Something the program needs is missing. The message explains what."""


class EmailError(Exception):
    """The email can't be processed. The message explains why."""


class DraftError(Exception):
    """A draft could not be stored. The message never includes server or exception text."""


# ---------------------------------------------------------------------------
# Reading an email
# ---------------------------------------------------------------------------


@dataclass
class IncomingEmail:
    sender: str
    sender_name: str
    subject: str
    body: str
    message_id: str
    references: list
    reply_to: str


def one_line(text):
    return " ".join(str(text).split())


def read_sender(msg):
    """Return the (display name, address) in the From header, or raise EmailError."""
    # The address decides whose balance can be read, so anything ambiguous is rejected.
    from_headers = msg.get_all("From", [])
    if len(from_headers) != 1:
        raise EmailError(f"Expected one From header but found {len(from_headers)}.")
    addresses = email.utils.getaddresses([str(from_headers[0])], strict=True)
    if len(addresses) != 1 or not re.fullmatch(r"[^@\s]+@[^@\s]+", addresses[0][1]):
        raise EmailError(f"Could not read one sender address from: {one_line(from_headers[0])}")
    return addresses[0]


def parse_email(raw):
    """Turn the bytes of an email into an IncomingEmail."""
    msg = email.message_from_bytes(raw, policy=email.policy.default)
    name, address = read_sender(msg)

    body_part = msg.get_body(preferencelist=("plain",))
    body = body_part.get_content().strip() if body_part else ""
    if not body:
        raise EmailError("The email has no plain text body.")

    message_ids = re.findall(r"<[^<>\s]+>", str(msg.get("Message-ID", "")))
    return IncomingEmail(
        sender=address,
        sender_name=one_line(name),
        subject=one_line(msg.get("Subject", "")),
        body=body,
        message_id=message_ids[0] if message_ids else "",
        references=re.findall(r"<[^<>\s]+>", str(msg.get("References", ""))),
        reply_to=address,
    )


def read_email_file(path):
    try:
        return parse_email(path.read_bytes())
    except OSError as e:
        raise EmailError(f"Could not read {path}: {e.strerror}.") from e


def reply_subject(subject):
    """Return "Re: subject", the subject Gmail expects for a reply in the same thread."""
    subject = re.sub(r"^(re:\s*)+", "", subject, flags=re.IGNORECASE)
    return f"Re: {subject}".strip()


# ---------------------------------------------------------------------------
# The save_reply tool
# ---------------------------------------------------------------------------


class DraftSaver:
    """Holds the one draft a run is allowed to save. Subclasses choose where it's stored."""

    tool = SAVE_REPLY_TOOL
    label = "Draft"
    status = "DRAFT, not sent"

    def __init__(self, incoming, source):
        self.incoming = incoming
        self.source = source
        self.saved_to = None
        self.text = None

    def save(self, tool_input):
        """Return (result text, is_error)."""
        if self.saved_to:
            return f"The reply was already saved to {self.saved_to}. Nothing new was saved.", True
        reply = tool_input.get("reply_text")
        review = tool_input.get("needs_human_review")
        if not isinstance(reply, str) or not reply.strip() or not isinstance(review, bool):
            return "save_reply needs non-empty reply_text and a true or false needs_human_review.", True

        reply = reply.strip()
        subject = reply_subject(self.incoming.subject)
        try:
            self.saved_to = self.store(subject, reply, self.render(subject, reply, review), review)
        except DraftError as e:
            return f"The draft could not be saved: {e}", True
        self.text = self.render(subject, reply, review)
        return self.result_message(), False

    def render(self, subject, reply, review):
        return (
            f"Status: {self.status}\n"
            f"To: {self.recipient_text()}\n"
            f"Subject: {subject}\n"
            f"Source email: {self.source}\n"
            f"Needs human review: {str(review).lower()}\n\n"
            f"{reply}\n"
        )

    def recipient_text(self):
        return self.incoming.sender

    def result_message(self):
        return f"Draft saved to {self.saved_to}. It has not been sent."

    def store(self, subject, reply, text, needs_review):
        """Store the draft and return where it went, or raise DraftError."""
        raise NotImplementedError


class FileDraftSaver(DraftSaver):
    """Writes the draft to a text file in week-04/drafts/."""

    def __init__(self, incoming, email_path):
        super().__init__(incoming, email_path.name)
        self.email_path = email_path

    def store(self, subject, reply, text, needs_review):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = DRAFTS_DIR / f"{self.email_path.stem}-{stamp}.txt"
        try:
            DRAFTS_DIR.mkdir(exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except OSError as e:
            raise DraftError(f"could not write the file ({type(e).__name__}).") from e
        return str(path)


class GmailDraftSaver(DraftSaver):
    """Creates a Gmail draft replying to the sender, in the same thread as their email.

    If a person must handle the email, the email itself also gets a Gmail label, which
    shows in the inbox list and can never go out as part of the reply.
    """

    def __init__(self, incoming, source, login, uid):
        super().__init__(incoming, source)
        self.login = login
        self.uid = uid

    def build_message(self, subject, reply, automatic=False):
        msg = email.message.EmailMessage()
        msg["From"] = self.login[0]
        # Always the From address, whose balance was read, and never a Reply-To address.
        msg["To"] = email.utils.formataddr((self.incoming.sender_name, self.incoming.reply_to))
        msg["Subject"] = subject
        msg["Date"] = email.utils.format_datetime(datetime.now().astimezone())
        msg["Message-ID"] = email.utils.make_msgid(domain=self.login[0].split("@")[-1])
        if self.incoming.message_id:
            msg["In-Reply-To"] = self.incoming.message_id
            msg["References"] = " ".join(self.incoming.references + [self.incoming.message_id])
        if automatic:
            # Tells other mail systems not to answer it automatically, which prevents reply loops.
            msg["Auto-Submitted"] = "auto-replied"
        msg.set_content(reply)
        return msg.as_bytes(policy=email.policy.SMTP)

    def store(self, subject, reply, text, needs_review):
        # APPEND only places a message in a mailbox. Nothing here can send it.
        try:
            message = self.build_message(subject, reply)
            with open_imap(self.login) as imap:
                # The label comes first: if it fails there's no draft, so the email is
                # reported as unanswered instead of waiting in Gmail with no flag.
                if needs_review:
                    add_review_label(imap, self.uid)
                mailbox = find_drafts_mailbox(imap)
                status, _ = imap.append(mailbox, r"(\Draft \Seen)", time.time(), message)
        except (imaplib.IMAP4.error, OSError, ValueError) as e:
            raise DraftError(f"Gmail reported a problem ({type(e).__name__}).") from e
        if status != "OK":
            raise DraftError("Gmail did not accept the draft.")
        return "the Gmail Drafts folder"


class WatchSaver(GmailDraftSaver):
    """For --watch: sends the reply if it doesn't need review, and otherwise keeps a draft.

    This is the only code in the program that can send email. Claude supplies only the text
    and the true or false value. The recipient is always the allowlisted sender.
    """

    tool = WATCH_SAVE_REPLY_TOOL

    def __init__(self, incoming, source, login, uid):
        super().__init__(incoming, source, login, uid)
        self.sent = False
        self.send_failed = False

    def recipient_text(self):
        return f"the allowlisted address of {self.incoming.sender}"

    def store(self, subject, reply, text, needs_review):
        if needs_review:
            return super().store(subject, reply, text, needs_review)
        try:
            send_reply(self.login, self.incoming.reply_to, self.build_message(subject, reply, automatic=True))
        except (smtplib.SMTPException, OSError, ValueError) as e:
            # It's never retried, in case the reply got through before the failure.
            self.send_failed = True
            print(f"Gmail SMTP reported a problem ({type(e).__name__}). Keeping a draft instead.\n")
            return super().store(subject, reply, text, needs_review)
        self.sent = True
        self.label = "Reply"
        self.status = "SENT"
        return "the sender through Gmail SMTP"

    def result_message(self):
        return "The reply was sent." if self.sent else super().result_message()


# ---------------------------------------------------------------------------
# Gmail over IMAP
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def open_imap(login):
    """Sign in on a new connection, which agent runs are too slow to keep waiting on."""
    # A default context verifies Gmail's certificate before the password is sent.
    context = ssl.create_default_context()
    with imaplib.IMAP4_SSL(load_inbox.IMAP_HOST, ssl_context=context, timeout=load_inbox.TIMEOUT_SECONDS) as imap:
        imap.login(*login)
        yield imap


def find_drafts_mailbox(imap):
    """Return the Drafts mailbox name, which depends on the account's language."""
    status, rows = imap.list()
    if status == "OK":
        for row in rows:
            match = re.fullmatch(rb'\(([^)]*)\)\s+"[^"]*"\s+(.+)', row) if isinstance(row, bytes) else None
            if match and b"\\drafts" in match.group(1).lower():
                return match.group(2).decode()
    raise DraftError("Gmail did not report a Drafts folder.")


def add_review_label(imap, uid):
    """Add the review label to an inbox email, creating the label if it's new."""
    label = f'"{REVIEW_LABEL}"'
    imap.create(label)  # Gmail answers NO if the label already exists, which is fine.
    status, _ = imap.select(load_inbox.MAILBOX)
    if status == "OK":
        status, _ = imap.uid("STORE", uid, "+X-GM-LABELS", f"({label})")
    if status != "OK":
        raise DraftError(f"Gmail did not add the {REVIEW_LABEL} label.")


def unread_uids(login):
    with open_imap(login) as imap:
        status, _ = imap.select(load_inbox.MAILBOX)
        if status != "OK":
            raise imaplib.IMAP4.error("select failed")
        status, data = imap.uid("SEARCH", None, "UNSEEN")
        if status != "OK":
            raise imaplib.IMAP4.error("search failed")
        return data[0].split()


def fetch_if_unread(imap, uid):
    """Return an email's bytes without marking it read, or None if it's already read."""
    status, _ = imap.select(load_inbox.MAILBOX)
    if status != "OK":
        raise imaplib.IMAP4.error("select failed")
    # PEEK leaves the email unread, so a failed download doesn't lose it.
    status, data = imap.uid("FETCH", uid, "(FLAGS BODY.PEEK[])")
    if status != "OK":
        raise imaplib.IMAP4.error("fetch failed")
    details, raw = b"", None
    for item in data:
        if isinstance(item, tuple):
            details += item[0]
            raw = item[1]
        elif item:
            details += item
    return None if b"\\seen" in details.lower() else raw


def peek_email(login, uid):
    with open_imap(login) as imap:
        return fetch_if_unread(imap, uid)


def claim_email(login, uid):
    """Mark an unread email as read and return its bytes, or None if it's already read."""
    with open_imap(login) as imap:
        raw = fetch_if_unread(imap, uid)
        if raw is None:
            return None
        status, _ = imap.uid("STORE", uid, "+FLAGS.SILENT", r"(\Seen)")
        if status != "OK":
            raise imaplib.IMAP4.error("store failed")
        return raw


def send_reply(login, recipient, message):
    """Send one message over Gmail SMTP. Gmail keeps a copy in the Sent folder."""
    # A default context verifies Gmail's certificate before the password is sent.
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context, timeout=load_inbox.TIMEOUT_SECONDS) as smtp:
        smtp.login(*login)
        smtp.sendmail(login[0], [recipient], message)


# ---------------------------------------------------------------------------
# One agent run
# ---------------------------------------------------------------------------


async def call_mcp_tool(session, name, tool_input):
    """Return (result text, is_error)."""
    result = await session.call_tool(name, tool_input)
    text = "\n".join(block.text for block in result.content if block.type == "text")
    return text, bool(result.is_error)


async def run_agent(claude, session, system_prompt, incoming, saver, usage):
    """Run the tool loop, adding to usage as it goes. Returns why it stopped."""
    listed = await session.list_tools()
    mcp_tool_names = {t.name for t in listed.tools}
    tools = [
        {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
        for t in listed.tools
    ] + [saver.tool]

    messages = [{
        "role": "user",
        "content": (
            "Prepare a reply to this employee email.\n\n"
            f"<email>\nFrom: {incoming.sender}\nSubject: {incoming.subject}\n\n{incoming.body}\n</email>"
        ),
    }]
    tool_calls = 0

    while True:
        response = await claude.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )
        usage["input"] += response.usage.input_tokens
        usage["output"] += response.usage.output_tokens

        if response.stop_reason != "tool_use":
            text = "".join(b.text for b in response.content if b.type == "text").strip()
            if text:
                print(f"Claude said: {text}\n")
            return f"Claude replied without a tool request (stop reason: {response.stop_reason})."

        # Sent back unchanged, including any thinking blocks.
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in (b for b in response.content if b.type == "tool_use"):
            if tool_calls == MAX_TOOL_CALLS:
                print(f"Not run, over the limit: {block.name} {json.dumps(block.input)}\n")
                continue
            tool_calls += 1
            print(f"Step {tool_calls}: {block.name}")
            print(f"  Inputs: {json.dumps(block.input, ensure_ascii=False)}")
            if block.name == "save_reply":
                text, is_error = saver.save(block.input)
            elif block.name in mcp_tool_names:
                text, is_error = await call_mcp_tool(session, block.name, block.input)
            else:
                text, is_error = f"Unknown tool: {block.name}", True
            print(f"  Result{' (error)' if is_error else ''}: {text[:RESULT_PREVIEW_CHARS]}\n")
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": text, "is_error": is_error})

        if tool_calls == MAX_TOOL_CALLS:
            return f"Reached the limit of {MAX_TOOL_CALLS} tool calls."
        messages.append({"role": "user", "content": results})


async def process_email(api_key, system_prompt, incoming, saver, usage, server_log):
    """Start the server for this sender and run the agent.

    Returns (why it stopped, whether the same failure would hit every other email too).
    """
    # The "=" form keeps the address from being read as a separate command-line option.
    # The server's environment has no API key or Gmail login; it needs neither.
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER), f"--as-employee={incoming.sender}"],
        cwd=str(HERE.parent),
    )
    async with anthropic.AsyncAnthropic(api_key=api_key) as claude, Client(stdio_client(server, errlog=server_log)) as session:
        try:
            return await run_agent(claude, session, system_prompt, incoming, saver, usage), False
        except anthropic.AuthenticationError:
            return f"Claude rejected the API key in {API_KEY_FILE.name}.", True
        except anthropic.NotFoundError:
            return f"Model {MODEL} was not found.", True
        except anthropic.RateLimitError:
            return "Rate limited by the API. Wait a moment and try again.", True
        except anthropic.APIStatusError as e:
            return f"API error {e.status_code}: {e.message}", True
        except anthropic.APIConnectionError:
            return "Could not reach the Claude API. Check your internet connection.", True


def run_one(api_key, system_prompt, incoming, saver):
    """Run the agent on one email and print the outcome.

    Returns (draft saved, the same failure would hit every other email too, usage).
    """
    print(f"Sender: {incoming.sender}\nSubject: {incoming.subject}\n")
    usage = {"input": 0, "output": 0}
    # The server's own log goes to a temporary file, so it can be printed (and saved) here.
    with tempfile.TemporaryFile() as server_log:
        try:
            reason, widespread = asyncio.run(process_email(api_key, system_prompt, incoming, saver, usage, server_log))
        except (Exception, BaseExceptionGroup) as e:  # the MCP client reports a failed server as a group
            if isinstance(e, BaseExceptionGroup) and e.subgroup(KeyboardInterrupt):
                raise KeyboardInterrupt from None  # Ctrl+C, wrapped by the MCP client
            reason, widespread = f"The HR tools server could not start or stopped responding ({type(e).__name__}).", True
        server_log.seek(0)
        log_text = server_log.read().decode("utf-8", errors="replace").strip()
    if log_text:
        print(f"HR tools server log:\n{log_text}\n")

    print(f"Stopped: {reason}\n")
    if saver.saved_to:
        print(f"{saver.label} ({saver.saved_to}):\n\n{saver.text}")
    else:
        print("No draft was saved.\n")
    print(f"Total tokens: {usage['input']} input, {usage['output']} output")
    return bool(saver.saved_to), widespread, usage


# ---------------------------------------------------------------------------
# The two ways to run
# ---------------------------------------------------------------------------


def run_email_file(path, api_key, system_prompt):
    try:
        incoming = read_email_file(path)
    except EmailError as e:
        print(f"Could not process the email: {e}")
        return 1
    print(f"Email: {path.name}")
    saved, _, _ = run_one(api_key, system_prompt, incoming, FileDraftSaver(incoming, path))
    return 0 if saved else 1


def run_inbox(api_key, system_prompt):
    try:
        login = load_inbox.read_login()
    except load_inbox.LoginFileError as e:
        print(e)
        return 1

    try:
        uids = unread_uids(login)
    except imaplib.IMAP4.error:
        print("Gmail rejected the login or the inbox request. No emails were marked as read.")
        return 1
    except OSError as e:
        print(f"Could not connect to Gmail ({type(e).__name__}). No emails were marked as read.")
        return 1
    if not uids:
        print("No unread emails in the inbox.")
        return 0
    print(f"Unread emails: {len(uids)}\n")

    drafted, no_draft = 0, []
    totals = {"input": 0, "output": 0}
    stopped_early = None
    for number, uid in enumerate(uids, start=1):
        label = f"UID {uid.decode()}"
        print(f"=== Email {number} of {len(uids)} ({label}) ===")
        # Claiming comes first, so this email can't be handled twice.
        try:
            raw = claim_email(login, uid)
        except (imaplib.IMAP4.error, OSError) as e:
            stopped_early = f"Gmail reported a problem while marking {label} as read ({type(e).__name__})."
            break
        if raw is None:
            print("Already read, so skipping it.\n")
            continue

        try:
            incoming = parse_email(raw)
        except EmailError as e:
            print(f"Could not process the email: {e}\n")
            no_draft.append(f"{label}: {e}")
            continue

        saver = GmailDraftSaver(incoming, f"inbox {label}", login, uid)
        saved, widespread, usage = run_one(api_key, system_prompt, incoming, saver)
        totals["input"] += usage["input"]
        totals["output"] += usage["output"]
        print()
        if saved:
            drafted += 1
        else:
            no_draft.append(f"{label}: \"{incoming.subject}\" from {incoming.sender}")
        if widespread:
            stopped_early = "The problem above would affect every email."
            break

    print("=== Summary ===")
    print(f"Drafts created: {drafted}")
    if no_draft:
        print(f"Marked as read but NOT answered ({len(no_draft)}). A person needs to handle these:")
        for item in no_draft:
            print(f"  {item}")
    if stopped_early:
        print(f"Stopped early: {stopped_early} Emails not reached are still unread.")
    print(f"Total tokens, all emails: {totals['input']} input, {totals['output']} output")
    print("Nothing was sent.")
    return 1 if no_draft or stopped_early else 0


def read_allowlist(account_address):
    """Return {real address, lowercase: employee address}. Any doubt stops the program.

    Error messages give line numbers only, never the file's contents.
    """
    try:
        text = ALLOWLIST_FILE.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise SetupError(f"{ALLOWLIST_FILE} was not found.") from None
    except (OSError, UnicodeError):
        raise SetupError(f"{ALLOWLIST_FILE.name} could not be read as text.") from None

    allowed = {}
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("=")]
        if len(parts) != 2 or not all(re.fullmatch(ADDRESS_PATTERN, part) for part in parts):
            raise SetupError(f"Line {number} of {ALLOWLIST_FILE.name} should read: real address = employee address")
        real = parts[0].lower()
        if allowed.get(real, parts[1]) != parts[1]:
            raise SetupError(f"Line {number} of {ALLOWLIST_FILE.name} repeats an address with a different employee.")
        if real == account_address.lower():
            # The agent would be answering its own replies.
            raise SetupError(f"Line {number} of {ALLOWLIST_FILE.name} lists the account the agent signs in to.")
        allowed[real] = parts[1]
    if not allowed:
        raise SetupError(f"{ALLOWLIST_FILE.name} has no addresses in it.")
    return allowed


def is_automatic(msg):
    """True for automatic mail: an Auto-Submitted header, or Precedence bulk or auto_reply."""
    if msg.get_all("Auto-Submitted"):
        return True
    precedence = {str(v).strip().lower().replace("-", "_") for v in msg.get_all("Precedence", [])}
    return bool(precedence & {"bulk", "auto_reply"})


def employee_for(raw, allowed):
    """Return the employee address an email may be handled as, or None to leave it unread."""
    msg = email.message_from_bytes(raw, policy=email.policy.default)
    if is_automatic(msg):
        return None
    try:
        return allowed.get(read_sender(msg)[1].lower())
    except EmailError:
        return None


class Redactor:
    """Hides secrets in everything printed, ahead of the screen and any --save file."""

    def __init__(self, output, secrets):
        self.output = output
        ordered = sorted(secrets, key=len, reverse=True)
        self.pattern = re.compile("|".join(re.escape(s) for s in ordered), re.IGNORECASE)

    def hide(self, text):
        return self.pattern.sub(HIDDEN, text)

    def write(self, text):
        return self.output.write(self.hide(text))

    def flush(self):
        self.output.flush()


class WatchRun:
    """Counts what a --watch run has done, for the rules and the summary."""

    def __init__(self):
        self.sent = 0
        self.drafted = 0
        self.no_draft = []
        self.left_unread = 0
        self.tokens = {"input": 0, "output": 0}


def handle_allowed_email(uid, employee, api_key, system_prompt, login, hide, run):
    """Handle one allowlisted email as its employee. Returns a reason to stop, or None."""
    label = f"UID {uid.decode()}"
    print(f"=== Email from an allowlisted sender ({label}) ===")
    raw = claim_email(login, uid)  # Marked as read first, so it can't be handled twice.
    if raw is None:
        print("Already read, so skipping it.\n")
        return None
    run.no_draft.append(label)  # Removed below once the email has an answer. Ctrl+C leaves it listed.
    try:
        parsed = parse_email(raw)
    except EmailError:
        print("Could not process the email.\n")
        return None

    # From here on the sender is the employee. Claude never sees the real address.
    incoming = replace(
        parsed,
        sender=employee,
        sender_name="",
        subject=hide(parsed.subject),
        body=hide(parsed.body),
    )
    saver = WatchSaver(incoming, f"inbox {label}", login, uid)
    _, widespread, usage = run_one(api_key, system_prompt, incoming, saver)
    print()
    run.tokens["input"] += usage["input"]
    run.tokens["output"] += usage["output"]
    if saver.saved_to:
        run.no_draft.remove(label)
    if saver.sent:
        run.sent += 1
    elif saver.saved_to:
        run.drafted += 1
    if saver.send_failed:
        return "Gmail SMTP failed, and sending is never retried automatically."
    if widespread:
        return "The problem above would affect every email."
    if run.sent >= MAX_SENT:
        return f"Reached the limit of {MAX_SENT} sent replies."
    return None


def watch_loop(api_key, system_prompt, login, allowed, hide):
    run = WatchRun()
    skipped = set()  # Emails left unread, so each is looked at and reported once.
    failures = 0
    print(
        f"Watching the inbox every {WATCH_SECONDS} seconds. Replies that don't need review are SENT, "
        f"up to {MAX_SENT}. Press Ctrl+C to stop.\n"
    )
    stop = None
    try:
        while stop is None:
            try:
                for uid in unread_uids(login):
                    if uid in skipped:
                        continue
                    raw = peek_email(login, uid)
                    if raw is None:
                        continue
                    employee = employee_for(raw, allowed)
                    if employee is None:
                        skipped.add(uid)
                        run.left_unread += 1
                        print(f"UID {uid.decode()}: automatic, or not from an allowlisted sender. Left unread.\n")
                        continue
                    stop = handle_allowed_email(uid, employee, api_key, system_prompt, login, hide, run)
                    if stop:
                        break
                failures = 0
            except (imaplib.IMAP4.abort, OSError) as e:
                # A network problem may pass, so the next check tries again.
                failures += 1
                print(f"Could not check Gmail ({type(e).__name__}), attempt {failures} of {MAX_CONNECTION_FAILURES}.\n")
                if failures >= MAX_CONNECTION_FAILURES:
                    stop = "Gmail could not be reached."
            except imaplib.IMAP4.error as e:
                stop = f"Gmail rejected the login or a request ({type(e).__name__})."
            if stop is None:
                time.sleep(WATCH_SECONDS)
    except KeyboardInterrupt:
        stop = "Ctrl+C."

    print("=== Summary ===")
    print(f"Stopped: {stop}")
    print(f"Replies sent: {run.sent} (limit {MAX_SENT})")
    print(f"Drafts kept: {run.drafted}")
    if run.no_draft:
        print(f"Marked as read but NOT answered ({len(run.no_draft)}). A person needs to handle these:")
        for item in run.no_draft:
            print(f"  {item}")
    print(f"Left unread and not sent to Claude: {run.left_unread}")
    print(f"Total tokens, all emails: {run.tokens['input']} input, {run.tokens['output']} output")
    return 1 if run.no_draft or stop != "Ctrl+C." and run.sent < MAX_SENT else 0


def run_watch(api_key, system_prompt):
    try:
        login = load_inbox.read_login()
        allowed = read_allowlist(login[0])
    except (load_inbox.LoginFileError, SetupError) as e:
        print(e)
        return 1

    screen = sys.stdout
    redactor = Redactor(screen, [login[1], login[0], *allowed])
    sys.stdout = redactor
    try:
        return watch_loop(api_key, system_prompt, login, allowed, redactor.hide)
    finally:
        sys.stdout = screen


# ---------------------------------------------------------------------------
# Program start
# ---------------------------------------------------------------------------


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


def load_api_key():
    try:
        key = API_KEY_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise SetupError(f"API key file not found: {API_KEY_FILE}") from None
    if not key:
        raise SetupError(f"API key file is empty: {API_KEY_FILE}")
    return key


def run(args):
    try:
        api_key = load_api_key()
        system_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")
    except SetupError as e:
        print(e)
        return 1
    except OSError:
        print(f"Could not read {SYSTEM_PROMPT_FILE}.")
        return 1
    if args.watch:
        return run_watch(api_key, system_prompt)
    if args.inbox:
        return run_inbox(api_key, system_prompt)
    return run_email_file(args.email, api_key, system_prompt)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Prepare replies to HR emails. Only --watch can send anything.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--email", type=Path, help="path of one email file; the draft goes to week-04/drafts/")
    source.add_argument("--inbox", action="store_true", help="handle each unread Gmail inbox email; drafts go to Gmail")
    source.add_argument("--watch", action="store_true",
                        help=f"check for unread email every {WATCH_SECONDS} seconds and SEND replies that don't "
                             f"need review, up to {MAX_SENT}, to allowlisted senders only")
    parser.add_argument("--save", type=Path, metavar="FILE",
                        help="also write everything printed to this UTF-8 text file, replacing it if it exists")
    args = parser.parse_args()

    if not args.save:
        sys.exit(run(args))

    # Opened first, so a bad path stops the program before any email is marked as read.
    try:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        save_file = args.save.open("w", encoding="utf-8")
    except OSError as e:
        sys.exit(f"Could not open {args.save} for writing ({type(e).__name__}).")
    screen = sys.stdout
    sys.stdout = Tee(screen, save_file)
    try:
        status = run(args)
    finally:
        sys.stdout = screen
        save_file.close()
    sys.exit(status)


if __name__ == "__main__":
    main()
