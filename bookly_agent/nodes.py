from dotenv import load_dotenv
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing import Literal, Optional
import json
import os

from state import BooklyAgentState
import tools as bookly_tools
import prompts
import samples_data.kb as kb

load_dotenv()

# Per-intent argument requirements. Used by gather_arguments and argument validation.
REQUIRED_ARGUMENTS: dict[str, list[str]] = {
    "order_status": ["order_id"],
    "return_refund": ["order_id", "postcode", "reason"],
    "general_question": [],
    "unknown": [],
}

# Max clarifying questions before we escalate on unknown intent.
# Prevents the agent from looping forever if classification keeps failing.
MAX_CLARIFICATION_ATTEMPTS = 2

# Public-facing help URL used when a general question isn't covered by the KB.
HELP_URL = "https://bookly.com/help"

base_llm = init_chat_model(model=os.getenv("LLM_MODEL"))


class IntentClassification(BaseModel):
    """Classification of the user's latest message into a support intent."""
    intent: Literal["order_status", "return_refund", "general_question", "unknown"] = Field(
        description="The classified intent for the latest user message."
    )

class ExtractedArguments(BaseModel):
    """Arguments extracted from the user's latest message."""
    order_id: Optional[str] = Field(default=None, description="4-digit order number, prefixes stripped.")
    postcode: Optional[str] = Field(default=None, description="UK postcode.")
    reason: Optional[str] = Field(default=None, description="Short phrase describing the return reason.")

class UnknownResponse(BaseModel):
    """Response for an unclassifiable user message — farewell vs. clarify."""
    is_farewell: bool = Field(description="True if the user is politely ending the conversation.")
    message: str = Field(description="The short message to send back to the user.")
####### NODE: CLASSIFY_INTENT
def classify_intent(state: BooklyAgentState) -> dict:
    """Classify the latest user message into one of the known intents.
 
    Runs on every user turn, so the user can switch intents mid-conversation.
    """

    classifier = base_llm.with_structured_output(IntentClassification)
    result = classifier.invoke([
        SystemMessage(content=prompts.CLASSIFY_INTENT_PROMPT),
        *state["messages"]
    ])
    intent = result.intent

    # Track unknown attempts so we can escalate after MAX_CLARIFICATION_ATTEMPTS.
    attempts = state.get("clarification_attempts", 0)
    if intent == "unknown":
        attempts += 1

    return {
        "intent": intent,
        "clarification_attempts": attempts,
    }


####### NODE: GATHER_ARGUMENTS
def gather_arguments(state: BooklyAgentState) -> dict:
    """Extract any required arguments from the user's latest message.
 
    Merges newly-extracted arguments into state.arguments, then computes missing_arguments.
    Does NOT ask the clarifying question — that's ask_clarification's job.
    """
    intent = state.get("intent")
    required = REQUIRED_ARGUMENTS.get(intent, [])
    current = state.get("arguments", {}).copy()

    prompt = prompts.GATHER_ARGUMENTS_PROMPT.format(
        intent=intent,
        required_arguments=required,
        current_arguments=current,
    )
    extractor = base_llm.with_structured_output(ExtractedArguments)
    result = extractor.invoke([
        SystemMessage(content=prompt),
        *state["messages"]
    ])
    extracted = result.model_dump(exclude_none=True)

    # Merge: new values overwrite old (user may correct a prior answer).
    current.update({k: v for k, v in extracted.items() if v})
 
    missing = [s for s in required if s not in current or not current[s]]
 
    return {
        "arguments": current,
        "missing_arguments": missing,
    }


####### NODE: ASK_CLARIFICATION
def ask_clarification(state: BooklyAgentState) -> dict:
    """Write a short clarifying question for the missing arguments."""
    prompt = prompts.ASK_CLARIFICATION_PROMPT.format(
        intent=state.get("intent"),
        missing_arguments=state.get("missing_arguments"),
    )
    response = base_llm.invoke([
        SystemMessage(content=prompt),
        *state["messages"],
    ])
    question = response.content.strip()
 
    return {
        "messages": [AIMessage(content=question)],
        "pending_response": question,
        "awaiting_user_input": True
    }


####### NODE: ASK_CLARIFICATION_UNKNOWN
def ask_clarification_unknown(state: BooklyAgentState) -> dict:
    """Handle unclassifiable messages: either say goodbye or ask what they need.

    A polite close ("no thanks", "bye") shouldn't be treated as a failed
    clarification — if we detect a farewell we reset the attempts counter
    so the user isn't penalized if they come back later in the session.
    """
    responder = base_llm.with_structured_output(UnknownResponse)
    result = responder.invoke([
        SystemMessage(content=prompts.ASK_CLARIFICATION_UNKNOWN_PROMPT),
        *state["messages"],
    ])

    update = {
        "messages": [AIMessage(content=result.message)],
        "pending_response": result.message,
        "awaiting_user_input": True,
    }
    if result.is_farewell:
        update["clarification_attempts"] = 0
    return update


def _invoke_tool(tool, args: dict) -> dict:
    """Run a tool and wrap the outcome in the standard tool_results entry shape."""
    try:
        result = tool.invoke(args)
        return {"tool": tool.name, "args": args, "result": result, "status": "ok"}
    except Exception as e:
        return {"tool": tool.name, "args": args, "error": str(e), "status": "error"}


####### NODE: EXECUTE_ACTION
def execute_action(state: BooklyAgentState) -> dict:
    """Perform the actual action for the current intent.

    - order_status: order_lookup
    - return_refund: issue_refund (eligibility and cap already checked upstream)

    Relies on the operator.add reducer on tool_results — returns only the new entry.
    """
    intent = state.get("intent")
    arguments = state.get("arguments", {})

    if intent == "order_status":
        try:
            args = {"order_id": arguments["order_id"]}
        except KeyError as e:
            entry = {"tool": bookly_tools.order_lookup.name, "args": {}, "error": f"missing argument: {e}", "status": "error"}
            return {"tool_results": [entry]}
        return {"tool_results": [_invoke_tool(bookly_tools.order_lookup, args)]}

    if intent == "return_refund":
        elig = state.get("eligibility_result") or {}
        # Topology guarantees we only reach here when eligibility passed; re-check
        # so a routing bug surfaces as a structured error instead of a silent £0 refund.
        if not elig.get("eligible") or "refund_amount" not in elig:
            entry = {
                "tool": bookly_tools.issue_refund.name,
                "args": {},
                "error": f"invalid eligibility_result: {elig!r}",
                "status": "error",
            }
            return {"tool_results": [entry]}
        try:
            args = {
                "order_id": arguments["order_id"],
                "amount": elig["refund_amount"],
                "identity_verified": state.get("identity_verified", False),
            }
        except KeyError as e:
            entry = {"tool": bookly_tools.issue_refund.name, "args": {}, "error": f"missing argument: {e}", "status": "error"}
            return {"tool_results": [entry]}
        return {"tool_results": [_invoke_tool(bookly_tools.issue_refund, args)]}

    raise ValueError(f"execute_action reached with unsupported intent: {intent!r}")

####### NODE: RESPOND
def respond(state: BooklyAgentState) -> dict:
    """Compose the final confirmation message based on tool results."""
    tool_results = state.get("tool_results", [])
    last_result = tool_results[-1] if tool_results else None

    prompt = prompts.RESPOND_PROMPT.format(
        intent=state.get("intent"),
        tool_results=json.dumps(last_result, default=str),
    )
    response = base_llm.invoke([
        SystemMessage(content=prompt),
        *state["messages"],
    ])
    message = response.content.strip()

    return {
        "messages": [AIMessage(content=message)],
        "pending_response": message,
        "awaiting_user_input": True,
    }

####### NODE: ANSWER_GENERAL
def answer_general(state: BooklyAgentState) -> dict:
    """Answer a general policy question by looking up relevant knowledge base topic(s).

    Uses the LLM to pick a topic, then grounds the response in the
    policy text returned by lookup_policy. If no topic matches, tells the
    user we don't have that info and redirects them to the website — no
    human escalation.
    """
    messages = state["messages"]

    # Step 1: Ask the LLM to pick the most relevant topic.
    picker = base_llm.invoke([
        SystemMessage(content=(
            "Pick the single most relevant Bookly policy topic for the user's question. "
            f"Available topics: {kb.AVAILABLE_POLICIES_TOPICS}. "
            "Respond with ONLY the topic name, or 'none' if nothing matches."
        )),
        *messages,
    ])
    topic = picker.content.strip().lower().strip('"\'')

    if topic == "none" or topic not in kb.AVAILABLE_POLICIES_TOPICS:
        message = (
            f"I don't have that information. You'll likely find it on our help "
            f"center: {HELP_URL}."
        )
        return {
            "messages": [AIMessage(content=message)],
            "pending_response": message,
            "awaiting_user_input": True,
        }

    lookup = bookly_tools.lookup_policy.invoke({"topic": topic})

    # Step 2: Compose a grounded answer. The policy text goes in the system
    # prompt as static context; the user's question comes via native messages.
    response = base_llm.invoke([
        SystemMessage(content=(
            f"{prompts.ANSWER_GENERAL_PROMPT}\n\n"
            f"If the policy text doesn't fully answer the question, say what you can "
            f"and direct the user to {HELP_URL} for anything else.\n\n"
            f"Relevant Bookly policy ({topic}):\n{lookup['content']}"
        )),
        *messages,
    ])
    message = response.content.strip()

    return {
        "messages": [AIMessage(content=message)],
        "pending_response": message,
        "awaiting_user_input": True,
        "tool_results": [{"tool": "lookup_policy", "args": {"topic": topic}, "result": lookup}],
    }


####### NODE: VERIFY_IDENTITY
def verify_identity(state: BooklyAgentState) -> dict:
    """Check that the postcode provided matches the postcode on file for the order (as a challenge).

    This is the topological gate: the graph cannot reach issue_refund
    without this node flipping identity_verified = True.
    """
    arguments = state.get("arguments", {})
    args = {
        "order_id": arguments.get("order_id", ""),
        "claimed_postcode": arguments.get("postcode", ""),
    }
    try:
        result = bookly_tools.verify_identity.invoke(args)
        entry = {"tool": bookly_tools.verify_identity.name, "args": args, "result": result, "status": "ok"}
    except Exception as e:
        result = {"verified": False, "reason": "tool_error"}
        entry = {"tool": bookly_tools.verify_identity.name, "args": args, "error": str(e), "status": "error"}

    return {
        "identity_verified": bool(result.get("verified")),
        "tool_results": [entry],
    }

####### NODE: RESPOND_IDENTITY_FAILED
def respond_identity_failed(state: BooklyAgentState) -> dict:
    """Terminal: identity verification failed. No escalation, no retry."""
    message = (
        "I couldn't verify your identity — the postcode you gave doesn't match "
        "the one on the order. Please double-check and try again."
    )
    return {
        "messages": [AIMessage(content=message)],
        "pending_response": message,
        "awaiting_user_input": True,
    }

####### NODE: CHECK_ELIGIBILITY
def check_eligibility(state: BooklyAgentState) -> dict:
    """Evaluate the refund against Bookly policy.

    Returns structured facts only. The graph's conditional edges decide
    what to do with the result (auto-approve under cap, escalate over cap,
    explain and decline if ineligible).
    """
    arguments = state.get("arguments", {})
    args = {
        "order_id": arguments.get("order_id", ""),
        "reason": arguments.get("reason", ""),
    }
    try:
        result = bookly_tools.check_return_eligibility.invoke(args)
        entry = {"tool": bookly_tools.check_return_eligibility.name, "args": args, "result": result, "status": "ok"}
    except Exception as e:
        result = {}
        entry = {"tool": bookly_tools.check_return_eligibility.name, "args": args, "error": str(e), "status": "error"}

    return {
        "eligibility_result": result,
        "tool_results": [entry],
    }

####### NODE: RESPOND_INELIGIBLE
def respond_ineligible(state: BooklyAgentState) -> dict:
    """Terminal: explain why the return isn't eligible. No escalation, no LLM."""
    elig = state.get("eligibility_result") or {}
    reason = elig.get("reason", "not_eligible")
 
    explanations = {
        "outside_return_window": (
            f"This order is outside our {elig.get('window_days', 30)}-day return "
            "window, so it's no longer eligible for a refund."
        ),
        "not_yet_delivered": (
            "This order hasn't been delivered yet — returns can be started once it arrives."
        ),
        "order_not_found": (
            "I couldn't find that order. Could you double-check the order number?"
        ),
    }
    message = explanations.get(reason, "This order isn't eligible for a return right now.")
 
    return {
        "messages": [AIMessage(content=message)],
        "pending_response": message,
        "awaiting_user_input": True,
    }


####### NODE: RESPOND_CANNOT_HELP
def respond_cannot_help(state: BooklyAgentState) -> dict:
    """Terminal: intent stayed unknown past MAX_CLARIFICATION_ATTEMPTS.

    No LLM, no human handoff — redirect to the help center and mark the
    conversation escalated so downstream systems can see why we gave up.
    """
    message = (
        "I'm not able to help with that — I handle order status, returns, "
        "refunds, and Bookly policy questions. For anything else, please "
        f"check {HELP_URL}."
    )
    return {
        "messages": [AIMessage(content=message)],
        "pending_response": message,
        "awaiting_user_input": True,
        "escalated": True,
        "escalation_reason": "max_clarification_attempts",
    }


####### CONDITIONAL EDGE HELPERS: used to decide the routing
#  After CLASSIFY_INTENT
def route_after_classify(state: BooklyAgentState) -> str:
    intent = state.get("intent")
    if intent == "order_status":
        return "gather_arguments"
    if intent == "return_refund":
        return "gather_arguments"
    if intent == "general_question":
        return "answer_general"
    # unknown
    if state.get("clarification_attempts", 0) >= MAX_CLARIFICATION_ATTEMPTS:
        return "respond_cannot_help"
    return "ask_clarification_unknown"

#  After GATHER_ARGUMENTS
def route_after_gather_arguments(state: BooklyAgentState) -> str:
    """After argument gathering, decide: ask user, verify identity, or execute."""
    if state.get("missing_arguments"):
        return "ask_clarification"
    intent = state.get("intent")
    if intent == "return_refund":
        return "verify_identity"
    # order_status and others go straight to action
    return "execute_action"

#  After VERIFY_IDENTITY
def route_after_verify_identity(state: BooklyAgentState) -> str:
    if state.get("identity_verified"):
        return "check_eligibility"
    return "respond_identity_failed"

#  After CHECK_ELIGIBILITY
def route_after_eligibility(state: BooklyAgentState) -> str:
    """After eligibility check, two branches: execute the refund or explain why not."""
    elig = state.get("eligibility_result") or {}
    if elig.get("eligible"):
        return "execute_action"
    return "respond_ineligible"