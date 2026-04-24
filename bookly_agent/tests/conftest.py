"""Pytest fixtures and the ScriptedLLM helper.

The bookly_agent modules use bare imports (`import nodes`, `import tools`,
`from state import ...`), so the bookly_agent directory must be on sys.path
for tests to import them. We do that here once.

The ScriptedLLM is a deterministic stand-in for `nodes.base_llm`. Each test
queues exactly the LLM responses the path under test will request, and
verifies routing/state without paying for or depending on a real model.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

BOOKLY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BOOKLY_DIR))


@dataclass
class _TextMsg:
    content: str


class ScriptedLLM:
    """Drop-in replacement for the chat model used by nodes.base_llm.

    Supports both call shapes the nodes use:
      - .with_structured_output(Schema).invoke(messages) -> Schema instance
      - .invoke(messages) -> object with .content (an AIMessage-like)

    Responses are consumed in the order they were queued. Tests should queue
    one entry per LLM call the path will make.
    """

    def __init__(self, responses: list | None = None):
        self.responses = list(responses or [])
        self.calls: list[tuple] = []

    def queue(self, *responses) -> "ScriptedLLM":
        self.responses.extend(responses)
        return self

    def _next(self, kind: str):
        if not self.responses:
            raise AssertionError(f"ScriptedLLM ran out of responses on {kind} call. Calls so far: {self.calls}")
        return self.responses.pop(0)

    def with_structured_output(self, schema):
        outer = self

        class _Bound:
            def invoke(self, messages):
                outer.calls.append(("structured", schema.__name__))
                value = outer._next(f"structured:{schema.__name__}")
                return schema(**value) if isinstance(value, dict) else value

        return _Bound()

    def invoke(self, messages):
        self.calls.append(("text",))
        value = self._next("text")
        return _TextMsg(content=value if isinstance(value, str) else str(value))


@pytest.fixture
def scripted_llm(monkeypatch):
    """Yield a fresh ScriptedLLM, monkeypatched onto nodes.base_llm."""
    import nodes
    fake = ScriptedLLM()
    monkeypatch.setattr(nodes, "base_llm", fake)
    return fake


@pytest.fixture
def app(scripted_llm):
    """Compiled graph with the scripted LLM in place.

    Node functions resolve `base_llm` from the `nodes` module namespace at
    call time, so the monkeypatch on `nodes.base_llm` is enough — the
    pre-compiled `agent.app` will pick up the fake without rebuilding.
    """
    import agent
    return agent.app
