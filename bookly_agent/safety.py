"""Input-boundary defenses against prompt injection.

The real safety in this agent comes from graph topology and structured outputs
(see agent.py, tools.py) — this layer just makes it harder for a malicious user
to smuggle instructions into the LLM's context by capping length, stripping
control chars, and neutralizing common jailbreak markers.
"""
from __future__ import annotations
import re

MAX_USER_INPUT_CHARS = 2000

# Patterns a naive chat model might mistake for a real role/instruction boundary.
# We neutralize matches by wrapping them in backticks so the model sees them as
# literal strings, not directives.
_INJECTION_MARKERS = re.compile(
    r"(?i)("
    r"<\s*/?\s*(?:system|assistant|user)\s*>"
    r"|\|im_start\||\|im_end\|"
    r"|\[(?:system|assistant|user)\]"
    r"|\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|earlier|above)\s+(?:instructions|messages|prompts|rules)"
    r"|\byou\s+are\s+now\s+"
    r"|\bnew\s+instructions\s*:"
    r"|\bsystem\s+prompt\s*:"
    r")"
)


def sanitize_user_input(text: str) -> str:
    """Cap length, strip control chars, and neutralize common injection markers.

    Run at the CLI/API boundary before wrapping the message in a HumanMessage.
    Preserves meaning for benign users; breaks the structure of typical
    jailbreak payloads.
    """
    if not isinstance(text, str):
        return ""
    # Drop ASCII control chars except newline and tab.
    text = "".join(ch for ch in text if ch in ("\n", "\t") or ord(ch) >= 0x20)
    text = _INJECTION_MARKERS.sub(lambda m: f"`{m.group(0)}`", text)
    if len(text) > MAX_USER_INPUT_CHARS:
        text = text[:MAX_USER_INPUT_CHARS]
    return text.strip()
