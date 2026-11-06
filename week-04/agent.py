"""Command-line agent that prepares replies to HR emails, using Claude's tool use.

Usage (from the top of the repository):
    .venv\\Scripts\\python.exe week-04\\agent.py --email week-04\\hartwell-hr\\sample-emails\\01-dana-vacation.eml
        Handles one email file and writes the draft to week-04/drafts/.
    .venv\\Scripts\\python.exe week-04\\agent.py --inbox
        Handles each unread email in the Gmail inbox and creates real Gmail drafts.
    Add --save output.txt to either one to also write everything printed to that file.

For each email, the agent starts hr_mcp_server.py with --as-employee and the sender's
address, so the balance tool can return only the sender's own balance. Claude gets that
server's tools plus save_reply. Each run stops after 10 tool calls, or on any reply from
Claude that isn't a tool request.

This program reads email and creates drafts only. It has no way to send email.

The Gmail login comes from Agent-Email.txt, in the folder above the repository, and is only
ever passed to Gmail. It is never printed or saved, so errors that might repeat it are
reported by type, never by their text.
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
import ssl
import sys
import tempfile
import time
from dataclasses import dataclass
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

MODEL = "claude-sonnet-5"
MAX_TOKENS = 16000
MAX_TOOL_CALLS = 10
RESULT_PREVIEW_CHARS = 300
REVIEW_LABEL = "NEEDS HR REVIEW"

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


def one_line(text):
    return " ".join(str(text).split())


def parse_email(raw):
    """Turn the bytes of an email into an IncomingEmail."""
    msg = email.message_from_bytes(raw, policy=email.policy.default)

    # The address decides whose balance can be read, so anything ambiguous is rejected.
    from_headers = msg.get_all("From", [])
    if len(from_headers) != 1:
        raise EmailError(f"Expected one From header but found {len(from_headers)}.")
    addresses = email.utils.getaddresses([str(from_headers[0])], strict=True)
    if len(addresses) != 1 or not re.fullmatch(r"[^@\s]+@[^@\s]+", addresses[0][1]):
        raise EmailError(f"Could not read one sender address from: {one_line(from_headers[0])}")

    body_part = msg.get_body(preferencelist=("plain",))
    body = body_part.get_content().strip() if body_part else ""
    if not body:
        raise EmailError("The email has no plain text body.")

    message_ids = re.findall(r"<[^<>\s]+>", str(msg.get("Message-ID", "")))
    return IncomingEmail(
        sender=addresses[0][1],
        sender_name=one_line(addresses[0][0]),
        subject=one_line(msg.get("Subject", "")),
        body=body,
        message_id=message_ids[0] if message_ids else "",
        references=re.findall(r"<[^<>\s]+>", str(msg.get("References", ""))),
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

    def __init__(self, incoming, source):
        self.incoming = incoming
        self.source = source
        self.saved_to = None
        self.text = None

    def save(self, tool_input):
        """Return (result text, is_error)."""
        if self.saved_to:
            return f"A draft was already saved to {self.saved_to}. Nothing new was saved.", True
        reply = tool_input.get("reply_text")
        review = tool_input.get("needs_human_review")
        if not isinstance(reply, str) or not reply.strip() or not isinstance(review, bool):
            return "save_reply needs non-empty reply_text and a true or false needs_human_review.", True

        reply = reply.strip()
        subject = reply_subject(self.incoming.subject)
        text = (
            f"Status: DRAFT, not sent\n"
            f"To: {self.incoming.sender}\n"
            f"Subject: {subject}\n"
            f"Source email: {self.source}\n"
            f"Needs human review: {str(review).lower()}\n\n"
            f"{reply}\n"
        )
        try:
            self.saved_to = self.store(subject, reply, text, review)
        except DraftError as e:
            return f"The draft could not be saved: {e}", True
        self.text = text
        return f"Draft saved to {self.saved_to}. It has not been sent.", False

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

    def build_message(self, subject, reply):
        msg = email.message.EmailMessage()
        msg["From"] = self.login[0]
        # Always the From address, whose balance was read, and never a Reply-To address.
        msg["To"] = email.utils.formataddr((self.incoming.sender_name, self.incoming.sender))
        msg["Subject"] = subject
        msg["Date"] = email.utils.format_datetime(datetime.now().astimezone())
        msg["Message-ID"] = email.utils.make_msgid(domain=self.login[0].split("@")[-1])
        if self.incoming.message_id:
            msg["In-Reply-To"] = self.incoming.message_id
            msg["References"] = " ".join(self.incoming.references + [self.incoming.message_id])
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


def claim_email(login, uid):
    """Mark an unread email as read and return its bytes, or None if it's already read."""
    with open_imap(login) as imap:
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
        if raw is None or b"\\seen" in details.lower():
            return None
        status, _ = imap.uid("STORE", uid, "+FLAGS.SILENT", r"(\Seen)")
        if status != "OK":
            raise imaplib.IMAP4.error("store failed")
        return raw


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
    ] + [SAVE_REPLY_TOOL]

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
        except Exception as e:  # the MCP client reports a failed server as an ExceptionGroup
            reason, widespread = f"The HR tools server could not start or stopped responding ({type(e).__name__}).", True
        server_log.seek(0)
        log_text = server_log.read().decode("utf-8", errors="replace").strip()
    if log_text:
        print(f"HR tools server log:\n{log_text}\n")

    print(f"Stopped: {reason}\n")
    if saver.saved_to:
        print(f"Draft ({saver.saved_to}):\n\n{saver.text}")
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
    if args.inbox:
        return run_inbox(api_key, system_prompt)
    return run_email_file(args.email, api_key, system_prompt)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Prepare draft replies to HR emails. Nothing is sent.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--email", type=Path, help="path of one email file; the draft goes to week-04/drafts/")
    source.add_argument("--inbox", action="store_true", help="handle each unread Gmail inbox email; drafts go to Gmail")
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
