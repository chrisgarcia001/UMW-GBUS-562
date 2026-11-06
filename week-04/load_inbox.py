"""Place the sample HR emails into a Gmail sandbox inbox as unread messages dated today.

Usage (from the top of the repository):
    .venv\\Scripts\\python.exe week-04\\load_inbox.py

Signs in over IMAP using the login in Agent-Email.txt, in the folder above the repository.
Messages are placed directly in the mailbox, so nothing is sent. Running it again adds
another copy of every message.

The login is only ever passed to Gmail. It is never printed or saved, and this script
never prints the text of a server error or exception in case it repeats any of the login.
"""

import email
import email.policy
import email.utils
import imaplib
import re
import ssl
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
EMAILS_DIR = HERE / "hartwell-hr" / "sample-emails"
LOGIN_FILE = HERE.parent.parent / "Agent-Email.txt"

IMAP_HOST = "imap.gmail.com"
MAILBOX = "INBOX"
TIMEOUT_SECONDS = 30

ADDRESS_LABELS = {"email", "address", "emailaddress", "user", "username", "login"}
PASSWORD_LABELS = {"password", "pass", "pwd", "apppassword", "gmailapppassword"}


class LoginFileError(Exception):
    """Agent-Email.txt can't be used. The message never includes the file's contents."""


def parse_login(text):
    """Return (address, password) from any of these layouts:

    EMAIL_ADDRESS=address and APP_PASSWORD=password on separate lines (":" also works)
    the address on one line and the password on another
    the address and password on one line, separated by a space, comma, colon, or semicolon
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    address = password = None

    for line in lines:
        match = re.fullmatch(r"([A-Za-z][A-Za-z _-]*?)\s*[=:]\s*(.+)", line)
        if match:
            label = re.sub(r"[ _-]", "", match.group(1)).lower()
            if label in ADDRESS_LABELS:
                address = match.group(2)
            elif label in PASSWORD_LABELS:
                password = match.group(2)

    if not (address and password) and len(lines) == 2:
        first, second = lines
        address, password = (first, second) if "@" in first else (second, first)
    elif not (address and password) and len(lines) == 1:
        match = re.fullmatch(r"(\S+@[^\s:,;]+)[\s:,;]+(.+)", lines[0])
        if match:
            address, password = match.groups()

    address = (address or "").strip().strip("\"'")
    # Google shows app passwords in groups separated by spaces, which aren't part of them.
    password = re.sub(r"\s", "", (password or "").strip().strip("\"'"))
    if not re.fullmatch(r"[^@\s]+@[^@\s]+", address) or not password:
        raise LoginFileError(
            f"Could not find an address and a password in {LOGIN_FILE.name}. Use two lines: "
            "EMAIL_ADDRESS=your address, then APP_PASSWORD=your app password."
        )
    return address, password


def read_login():
    try:
        return parse_login(LOGIN_FILE.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise LoginFileError(f"{LOGIN_FILE} was not found.") from None
    except (OSError, UnicodeError):
        raise LoginFileError(f"{LOGIN_FILE.name} could not be read as text.") from None


def prepare_message(path):
    """Return the email as bytes for IMAP, with today's date and its own Message-ID."""
    msg = email.message_from_bytes(path.read_bytes(), policy=email.policy.default)
    del msg["Date"]
    del msg["Message-ID"]
    msg["Date"] = email.utils.format_datetime(datetime.now().astimezone())
    msg["Message-ID"] = email.utils.make_msgid(domain="hartwellfoods.example")
    return msg.as_bytes(policy=email.policy.SMTP)


def load_messages(address, password, messages):
    """Place the messages in the inbox. Returns the exit status for the program."""
    # A default context verifies Gmail's certificate before the password is sent.
    # imaplib doesn't do that check unless it's given one.
    with imaplib.IMAP4_SSL(IMAP_HOST, ssl_context=ssl.create_default_context(), timeout=TIMEOUT_SECONDS) as imap:
        try:
            imap.login(address, password)
        except imaplib.IMAP4.error:
            print(
                "Gmail rejected the login. Nothing was placed in the inbox. Check that "
                f"{LOGIN_FILE.name} has the right address and a Google app password, not the "
                "account password."
            )
            return 1

        loaded = 0
        for name, message in messages:
            # No flags means no \\Seen flag, so the message is unread.
            status, _ = imap.append(MAILBOX, None, imaplib.Time2Internaldate(time.time()), message)
            if status != "OK":
                print(f"Gmail did not accept {name}. Stopping after {loaded} of {len(messages)} messages.")
                return 1
            loaded += 1
            print(f"Placed {name} in {MAILBOX} as unread.")

        print(f"Done: placed {loaded} messages. Nothing was sent.")
        return 0


def main():
    paths = sorted(EMAILS_DIR.glob("*.eml"))
    if not paths:
        sys.exit(f"No .eml files found in {EMAILS_DIR}.")
    try:
        messages = [(p.name, prepare_message(p)) for p in paths]
        address, password = read_login()
    except LoginFileError as e:
        sys.exit(str(e))
    except OSError as e:
        sys.exit(f"Could not read the sample emails ({type(e).__name__}).")

    try:
        status = load_messages(address, password, messages)
    except ssl.SSLError:
        status = "Could not verify a secure connection to Gmail, so the login was not sent."
    except OSError:
        status = "Could not reach Gmail. Check your internet connection."
    except imaplib.IMAP4.error as e:
        status = f"Gmail reported a problem ({type(e).__name__}). Check the inbox before running this again."
    sys.exit(status)


if __name__ == "__main__":
    main()
