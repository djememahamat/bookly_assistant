"""Knowledge base loader.

Parses policies.md at import time into a topic-keyed dict. This makes
the markdown file the single source of truth — editing a policy in the
markdown updates the agent's knowledge with no code change.

Parsing rules
-------------
- Every lookup-able policy is an H3 section (### heading).
- Each H3 heading is followed by an HTML comment declaring the slug:

      ### Return window
      <!-- topic: return_window -->
      ...content...

- H3 sections without a topic comment are silently ignored (this lets
  authors add informational H3s that aren't agent-facing).
- Content accumulates until the next H3, next H2, or EOF.
"""

from __future__ import annotations
import re
from pathlib import Path


_TOPIC_RE = re.compile(r'<!--\s*topic:\s*([\w_-]+)\s*-->')


def parse_policies(markdown: str) -> dict[str, str]:
    """Split markdown into a {topic_slug: content} dict.

    Walks each H3 section, extracts the topic slug from the HTML comment
    immediately following the heading, and captures the body text up to
    the next H3/H2 boundary.
    """
    # Split on H3 boundaries, keeping the delimiter (?=...) so each chunk
    # starts with its own heading. Discard anything before the first H3.
    sections = re.split(r'^(?=###\s)', markdown, flags=re.MULTILINE)
    topics: dict[str, str] = {}

    for section in sections:
        if not section.startswith('###'):
            continue

        topic_match = _TOPIC_RE.search(section)
        if not topic_match:
            # H3 without a topic pragma — skip silently.
            continue

        slug = topic_match.group(1)

        # Body = everything after the topic comment, trimmed at next H2 if any.
        body = section[topic_match.end():]
        body = re.split(r'^##\s', body, maxsplit=1, flags=re.MULTILINE)[0]
        # Strip trailing horizontal rules (---) that may precede a section break.
        body = re.sub(r'\n\s*---\s*$', '', body.strip())
        topics[slug] = body.strip()

    return topics


def _default_path() -> Path:
    return Path(__file__).parent.parent / "samples_data" / "policies.md"


# Module-level singletons, loaded once at import time.
# If policies.md is missing or unreadable, this raises loudly — we want a
# clear failure, not a silently-empty KB.
POLICIES: dict[str, str] = parse_policies(_default_path().read_text())
AVAILABLE_POLICIES_TOPICS: list[str] = list(POLICIES.keys())

# print("AVAILABLE_POLICIES_TOPICS:", AVAILABLE_POLICIES_TOPICS)

def lookup(topic: str) -> str | None:
    """Return the policy content for a topic, or None if unknown."""
    return POLICIES.get(topic)