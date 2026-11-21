"""Alder Creek service desk assistant.

Sends a customer's service request to Claude with the assistant prompt and the
service guide as the system prompt, and prints Claude's JSON reply.

    python assistant.py --request "Our heat isn't working."
"""

import argparse
import json
import re
import sys
from pathlib import Path

import anthropic

HERE = Path(__file__).resolve().parent
PROMPT_FILE = HERE / "alder-creek" / "assistant-prompt.txt"
GUIDE_FILE = HERE / "alder-creek" / "service-guide.md"
# API-Key.txt lives in the folder that contains the repository (never inside it).
API_KEY_FILE = HERE.parent.parent / "API-Key.txt"

MODEL = "claude-opus-5-5"
MAX_TOKENS = 16000

# The reply must match this shape (structured outputs), so it is always valid JSON.
REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "priority": {"type": "string", "enum": ["Safety", "Emergency", "Urgent", "Routine"]},
        "reply": {"type": "string"},
        "needs_person": {"type": "boolean"},
    },
    "required": ["priority", "reply", "needs_person"],
    "additionalProperties": False,
}


# --- Guardrails (enforced in code, not in the prompt) ---

# 1. Before calling Claude: safety words. Matched as whole words, ignoring capital letters.
SAFETY_RE = re.compile(
    r"\bgas\b|\bcarbon monoxide\b|\bsmok(?:e|es|ed|ing|y)\b"
    r"|\bburning smells?\b|\bsmell(?:s|ed|ing)? (?:of |like )?burning\b",
    re.IGNORECASE,
)
SAFETY_REPLY = {
    "priority": "Safety",
    "reply": (
        "Please leave the building now and call the gas company or 911 from outside. "
        "Someone from Alder Creek will call you right away."
    ),
    "needs_person": True,
}

# 2. After Claude replies: things the desk must never say on its own.
ALLOWED_AMOUNTS = {0, 89, 149, 129}
# A dollar amount such as $89, $1,200 or $89.00.
DOLLAR_RE = re.compile(r"\$\s*(\d[\d,]*(?:\.\d+)?)")
BOOKING_RE = re.compile(r"\b(?:booked|confirmed)\b", re.IGNORECASE)
WARRANTY_RE = re.compile(r"\b(?:covered|under warranty)\b", re.IGNORECASE)
# A clock time such as 10 am, 10:30, 2 p.m. or 6pm.
CLOCK_RE = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)(?![a-z])|\b\d{1,2}:\d{2}\b", re.IGNORECASE)


def unexpected_amounts(text):
    """Dollar amounts in the text that aren't $0, $89, $149 or $129."""
    return [f"${m}" for m in DOLLAR_RE.findall(text) if float(m.replace(",", "")) not in ALLOWED_AMOUNTS]


def hold_reasons(text):
    """Why a reply must be read by a person before the customer sees it (empty if it needn't be)."""
    reasons = []
    amounts = unexpected_amounts(text)
    if amounts:
        reasons.append("dollar amount not in the fee list: " + ", ".join(amounts))
    words = sorted({m.lower() for m in BOOKING_RE.findall(text)})
    if words:
        reasons.append("says " + " and ".join(f'"{w}"' for w in words))
    warranty = sorted({m.lower() for m in WARRANTY_RE.findall(text)})
    if warranty:
        reasons.append("talks about coverage: " + ", ".join(f'"{w}"' for w in warranty))
    times = CLOCK_RE.findall(text)
    if times:
        reasons.append("gives a clock time: " + ", ".join(t.strip() for t in times))
    return reasons


class AssistantError(Exception):
    pass


def load_api_key():
    try:
        key = API_KEY_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise AssistantError(f"API key file not found: {API_KEY_FILE}") from None
    if not key:
        raise AssistantError(f"API key file is empty: {API_KEY_FILE}")
    return key


def build_system_prompt():
    """The assistant prompt, followed by the service guide."""
    try:
        prompt = PROMPT_FILE.read_text(encoding="utf-8").strip()
        guide = GUIDE_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError as e:
        raise AssistantError(f"Could not read {e.filename}") from None
    return f"{prompt}\n\n{guide}"


def assist(request, client=None):
    """Send one service request to Claude and return its reply as a dict.

    Safety requests get a fixed reply without calling Claude. A reply that needs
    checking gets needs_person = true and a held_because field giving the reason.
    """
    if SAFETY_RE.search(request):
        return dict(SAFETY_REPLY)

    client = client or anthropic.Anthropic(api_key=load_api_key())
    try:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=build_system_prompt(),
            messages=[{"role": "user", "content": request}],
            output_config={"format": {"type": "json_schema", "schema": REPLY_SCHEMA}},
            # If Claude declines for safety reasons, the API retries on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.AuthenticationError:
        raise AssistantError(f"Claude rejected the API key in {API_KEY_FILE.name}.") from None
    except anthropic.NotFoundError:
        raise AssistantError(f"Model {MODEL} was not found.") from None
    except anthropic.RateLimitError:
        raise AssistantError("Rate limited by the API. Wait a moment and try again.") from None
    except anthropic.APIStatusError as e:
        raise AssistantError(f"API error {e.status_code}: {e.message}") from None
    except anthropic.APIConnectionError:
        raise AssistantError("Could not reach the Claude API. Check your internet connection.") from None

    if response.stop_reason == "refusal":
        raise AssistantError("Claude declined to answer this request.")
    if response.stop_reason == "max_tokens":
        raise AssistantError("The reply was cut off (max_tokens reached).")
    text = "".join(block.text for block in response.content if block.type == "text")
    try:
        reply = json.loads(text)
    except json.JSONDecodeError:
        raise AssistantError(f"Claude's reply was not valid JSON:\n{text}") from None

    reasons = hold_reasons(reply["reply"])
    if reasons:
        reply["needs_person"] = True
        reply["held_because"] = "; ".join(reasons)
    return reply


def main():
    parser = argparse.ArgumentParser(description="Alder Creek service desk assistant.")
    parser.add_argument("--request", required=True, help="the customer's service request text")
    args = parser.parse_args()
    if not args.request.strip():
        parser.error("--request is empty")
    try:
        reply = assist(args.request)
    except AssistantError as e:
        print(e, file=sys.stderr)
        return 1
    print(json.dumps(reply, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
