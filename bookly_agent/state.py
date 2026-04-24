import operator
from typing import TypedDict, Optional, Literal, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


Intent = Literal["order_status", "return_refund", "general_question", "unknown"]
# Intent = Literal["order_status", "general_question", "unknown"]


class BooklyAgentState(TypedDict, total=False):
    # --- Conversation history ---
    # add_messages reducer: returning [msg] from a node appends instead of replacing.
    messages: Annotated[list[BaseMessage], add_messages]

    # --- Routing ---
    intent: Optional[Intent]
    clarification_attempts: int  # increments when intent=unknown; escalate at MAX

    # --- Arguments (progressively filled across turns) ---
    arguments: dict  # e.g. {"order_id": "1042", "email": "jesse@...", "reason": "damaged"}
    missing_arguments: list[str]

    # --- Gates (flip once, never back) ---
    identity_verified: bool

    # --- Policy check results ---
    eligibility_result: Optional[dict]
    # Shape: {"eligible": bool, "reason": str, "refund_amount": float}

    # --- Tool call log  ---
    # operator.add reducer: nodes return only the new entries, list is concatenated.
    tool_results: Annotated[list[dict], operator.add]
    # Each entry: {"tool": "order_lookup", "args": {...}, "result": {...}, "status": "ok"}

    # --- Control flow signals to the outer loop ---
    pending_response: Optional[str]  # message to emit to the user
    awaiting_user_input: bool        # if True, the outer loop pauses for input
    escalated: bool
    escalation_reason: Optional[str]
    escalation_ticket: Optional[dict]


def initial_state() -> BooklyAgentState:
    """Fresh state for a new conversation."""
    return {
        "messages": [],
        "intent": None,
        "clarification_attempts": 0,
        "arguments": {},
        "missing_arguments": [],
        # "identity_verified": False,
        # "eligibility_result": None,
        "tool_results": [],
        "pending_response": None,
        "awaiting_user_input": False,
        "escalated": False,
        "escalation_reason": None,
        "escalation_ticket": None,
    }