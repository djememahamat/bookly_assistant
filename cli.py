"""Interactive CLI for the Bookly agent.

Multi-turn chat: state (arguments, identity_verified, eligibility_result,
clarification_attempts, ...) persists across turns via a MemorySaver
checkpointer keyed by thread_id.

Run:
    uv run python cli.py
    uv run python cli.py --thread my-session   # custom thread id

Commands inside the REPL:
    /reset    start a new conversation (new thread id)
    /state    dump the current graph state
    /quit     exit
"""
from __future__ import annotations
import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "bookly_agent"))

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from agent import builder
from utils.safety import sanitize_user_input


def make_app():
    """Compile the graph with a MemorySaver so turns share state by thread_id."""
    return builder.compile(checkpointer=MemorySaver())


def run(thread_id: str) -> None:
    app = make_app()
    config = {"configurable": {"thread_id": thread_id}}

    print(f"Bookly agent CLI — thread {thread_id}")
    print("Type /quit to exit, /reset for a new thread, /state to dump state.\n")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not user:
            continue
        if user in ("/quit", "/exit"):
            return
        if user == "/reset":
            thread_id = uuid.uuid4().hex[:8]
            config = {"configurable": {"thread_id": thread_id}}
            print(f"--- new thread: {thread_id} ---\n")
            continue
        if user == "/state":
            snapshot = app.get_state(config).values
            for k in ("intent", "arguments", "missing_arguments", "identity_verified", "eligibility_result", "clarification_attempts"):
                if k in snapshot:
                    print(f"  {k}: {snapshot[k]}")
            print()
            continue

        safe_user = sanitize_user_input(user)
        if not safe_user:
            print("agent> Sorry, I didn't catch that — could you rephrase?\n")
            continue
        result = app.invoke({"messages": [HumanMessage(content=safe_user)]}, config=config)
        print(f"agent> {result['messages'][-1].content}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", default=uuid.uuid4().hex[:8], help="conversation thread id")
    args = parser.parse_args()
    run(args.thread)


if __name__ == "__main__":
    main()
