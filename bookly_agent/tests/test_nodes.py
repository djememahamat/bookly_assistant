"""L2 — Node unit tests.

Non-LLM nodes are called directly. LLM nodes use the scripted LLM fixture
to guarantee deterministic behavior."""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

import nodes


# ---------- execute_action (no LLM) ----------

def test_execute_action_order_status_happy():
    state = {"intent": "order_status", "arguments": {"order_id": "1042"}}
    out = nodes.execute_action(state)
    entry = out["tool_results"][0]
    assert entry["tool"] == "order_lookup"
    assert entry["status"] == "ok"
    assert entry["result"]["found"] is True


def test_execute_action_return_refund_happy():
    state = {
        "intent": "return_refund",
        "arguments": {"order_id": "1043"},
        "eligibility_result": {"eligible": True, "refund_amount": 14.99},
        "identity_verified": True,
    }
    out = nodes.execute_action(state)
    entry = out["tool_results"][0]
    assert entry["tool"] == "issue_refund"
    assert entry["status"] == "ok"
    assert entry["result"]["success"] is True
    assert entry["result"]["amount"] == 14.99


def test_execute_action_return_refund_blocked_when_eligibility_missing():
    """Defensive guard added in the recent refactor: if routing somehow
    sends an ineligible request to execute_action, fail loudly with a
    structured error rather than silently issuing a £0 refund."""
    state = {
        "intent": "return_refund",
        "arguments": {"order_id": "1043"},
        "eligibility_result": {},  # missing
        "identity_verified": True,
    }
    out = nodes.execute_action(state)
    entry = out["tool_results"][0]
    assert entry["status"] == "error"
    assert "invalid eligibility_result" in entry["error"]


def test_execute_action_return_refund_blocked_when_eligible_false():
    state = {
        "intent": "return_refund",
        "arguments": {"order_id": "1043"},
        "eligibility_result": {"eligible": False, "refund_amount": 14.99},
        "identity_verified": True,
    }
    out = nodes.execute_action(state)
    assert out["tool_results"][0]["status"] == "error"


def test_execute_action_order_status_missing_order_id_returns_error_entry():
    """KeyError path now produces a structured error instead of crashing."""
    state = {"intent": "order_status", "arguments": {}}
    out = nodes.execute_action(state)
    entry = out["tool_results"][0]
    assert entry["status"] == "error"
    assert "missing argument" in entry["error"]


def test_execute_action_return_refund_missing_order_id_returns_error_entry():
    state = {
        "intent": "return_refund",
        "arguments": {},  # no order_id
        "eligibility_result": {"eligible": True, "refund_amount": 14.99},
        "identity_verified": True,
    }
    out = nodes.execute_action(state)
    assert out["tool_results"][0]["status"] == "error"


def test_execute_action_unsupported_intent_raises():
    with pytest.raises(ValueError, match="unsupported intent"):
        nodes.execute_action({"intent": "general_question", "arguments": {}})


# ---------- verify_identity node (no LLM) ----------

def test_verify_identity_node_match():
    state = {"arguments": {"order_id": "1043", "postcode": "M13 9PL"}}
    out = nodes.verify_identity(state)
    assert out["identity_verified"] is True
    assert out["tool_results"][0]["status"] == "ok"


def test_verify_identity_node_mismatch():
    state = {"arguments": {"order_id": "1043", "postcode": "WRONG"}}
    out = nodes.verify_identity(state)
    assert out["identity_verified"] is False


def test_verify_identity_node_missing_arguments_doesnt_crash():
    """Empty arguments shouldn't crash the node — verify_identity handles it."""
    out = nodes.verify_identity({"arguments": {}})
    assert out["identity_verified"] is False


# ---------- check_eligibility node (no LLM) ----------

def test_check_eligibility_node_eligible():
    state = {"arguments": {"order_id": "1043", "reason": "damaged"}}
    out = nodes.check_eligibility(state)
    assert out["eligibility_result"]["eligible"] is True
    assert out["eligibility_result"]["refund_amount"] == 14.99


def test_check_eligibility_node_ineligible_outside_window():
    state = {"arguments": {"order_id": "1045", "reason": "damaged"}}
    out = nodes.check_eligibility(state)
    assert out["eligibility_result"]["eligible"] is False
    assert out["eligibility_result"]["reason"] == "outside_return_window"


# ---------- respond_ineligible (no LLM) ----------

@pytest.mark.parametrize(
    "reason,expected_substring",
    [
        ("outside_return_window", "30-day return"),
        ("not_yet_delivered", "hasn't been delivered"),
        ("order_not_found", "double-check"),
        ("some_unknown_reason", "isn't eligible"),
    ],
)
def test_respond_ineligible_per_reason(reason, expected_substring):
    state = {"eligibility_result": {"reason": reason, "window_days": 30}}
    out = nodes.respond_ineligible(state)
    assert expected_substring in out["pending_response"]
    assert out["awaiting_user_input"] is True


# ---------- respond_identity_failed (no LLM) ----------

def test_respond_identity_failed_message():
    out = nodes.respond_identity_failed({})
    assert "couldn't verify your identity" in out["pending_response"]
    assert out["awaiting_user_input"] is True


# ---------- classify_intent (LLM) ----------

@pytest.mark.parametrize("intent", ["order_status", "return_refund", "general_question", "unknown"])
def test_classify_intent_each_label(scripted_llm, intent):
    scripted_llm.queue({"intent": intent})
    state = {"messages": [HumanMessage(content="anything")]}
    out = nodes.classify_intent(state)
    assert out["intent"] == intent


def test_classify_intent_unknown_increments_counter(scripted_llm):
    scripted_llm.queue({"intent": "unknown"})
    state = {"messages": [HumanMessage(content="???")], "clarification_attempts": 1}
    out = nodes.classify_intent(state)
    assert out["clarification_attempts"] == 2


def test_classify_intent_known_resets_counter(scripted_llm):
    """A successful classification clears any accumulated unknown count so
    an isolated farewell later in the conversation still reaches
    ask_clarification_unknown instead of being escalated."""
    scripted_llm.queue({"intent": "order_status"})
    state = {"messages": [HumanMessage(content="where is my order")], "clarification_attempts": 1}
    out = nodes.classify_intent(state)
    assert out["clarification_attempts"] == 0


# ---------- gather_arguments (LLM) ----------

def test_gather_arguments_extracts_all_required(scripted_llm):
    scripted_llm.queue({"order_id": "1043", "postcode": "M13 9PL", "reason": "damaged"})
    state = {
        "intent": "return_refund",
        "messages": [HumanMessage(content="refund order 1043 postcode M13 9PL damaged")],
        "arguments": {},
    }
    out = nodes.gather_arguments(state)
    assert out["arguments"] == {"order_id": "1043", "postcode": "M13 9PL", "reason": "damaged"}
    assert out["missing_arguments"] == []


def test_gather_arguments_reports_missing(scripted_llm):
    scripted_llm.queue({"order_id": "1043"})  # only order_id, postcode + reason missing
    state = {
        "intent": "return_refund",
        "messages": [HumanMessage(content="refund order 1043")],
        "arguments": {},
    }
    out = nodes.gather_arguments(state)
    assert out["arguments"] == {"order_id": "1043"}
    assert set(out["missing_arguments"]) == {"postcode", "reason"}


def test_gather_arguments_merges_with_prior(scripted_llm):
    """User answers a follow-up; new value should merge with prior arguments."""
    scripted_llm.queue({"postcode": "M13 9PL"})
    state = {
        "intent": "return_refund",
        "messages": [HumanMessage(content="M13 9PL")],
        "arguments": {"order_id": "1043", "reason": "damaged"},
    }
    out = nodes.gather_arguments(state)
    assert out["arguments"] == {"order_id": "1043", "reason": "damaged", "postcode": "M13 9PL"}
    assert out["missing_arguments"] == []


def test_gather_arguments_new_value_overwrites_prior(scripted_llm):
    """User corrects themselves — newer extraction wins."""
    scripted_llm.queue({"order_id": "1044"})
    state = {
        "intent": "return_refund",
        "messages": [HumanMessage(content="actually 1044")],
        "arguments": {"order_id": "1043", "postcode": "M13 9PL", "reason": "damaged"},
    }
    out = nodes.gather_arguments(state)
    assert out["arguments"]["order_id"] == "1044"


# ---------- ask_clarification (LLM) ----------

def test_ask_clarification_emits_question(scripted_llm):
    scripted_llm.queue("What's the order number and your postcode?")
    state = {"intent": "return_refund", "missing_arguments": ["order_id", "postcode"], "messages": []}
    out = nodes.ask_clarification(state)
    assert out["pending_response"] == "What's the order number and your postcode?"
    assert out["awaiting_user_input"] is True
    assert out["messages"][0].content == "What's the order number and your postcode?"


# ---------- respond (LLM) ----------

def test_respond_composes_from_tool_results(scripted_llm):
    scripted_llm.queue("Refund of £14.99 issued for order 1043. RF-ABCD1234.")
    state = {
        "intent": "return_refund",
        "tool_results": [{"tool": "issue_refund", "result": {"success": True, "amount": 14.99, "currency": "GBP"}}],
        "messages": [],
    }
    out = nodes.respond(state)
    assert "£14.99" in out["pending_response"]
    assert out["awaiting_user_input"] is True


# ---------- answer_general (LLM) ----------

def test_answer_general_kb_hit(scripted_llm):
    # First call: topic picker. Second call: grounded answer.
    scripted_llm.queue("uk_delivery_time", "UK orders arrive in 2-4 working days from dispatch.")
    state = {"messages": [HumanMessage(content="how long does delivery take?")]}
    out = nodes.answer_general(state)
    assert "2-4 working days" in out["pending_response"]
    assert out["tool_results"][0]["tool"] == "lookup_policy"
    assert out["tool_results"][0]["args"]["topic"] == "uk_delivery_time"


def test_answer_general_no_topic_match_redirects_to_help_url(scripted_llm):
    scripted_llm.queue("none")  # picker returns "none"
    state = {"messages": [HumanMessage(content="why is the sky blue?")]}
    out = nodes.answer_general(state)
    assert nodes.HELP_URL in out["pending_response"]


def test_answer_general_unknown_topic_also_redirects(scripted_llm):
    scripted_llm.queue("not_a_real_topic")
    state = {"messages": [HumanMessage(content="?")]}
    out = nodes.answer_general(state)
    assert nodes.HELP_URL in out["pending_response"]


# ---------- ask_clarification_unknown (LLM, structured) ----------

def test_ask_clarification_unknown_farewell_resets_counter(scripted_llm):
    """A polite close resets clarification_attempts so the user isn't
    penalized if they come back later in the session."""
    scripted_llm.queue({"is_farewell": True, "message": "Thanks, take care!"})
    state = {
        "messages": [HumanMessage(content="thanks")],
        "clarification_attempts": 2,
    }
    out = nodes.ask_clarification_unknown(state)
    assert out["clarification_attempts"] == 0
    assert "escalated" not in out
    assert "Thanks" in out["pending_response"]


def test_ask_clarification_unknown_farewell_runs_even_at_threshold(scripted_llm):
    """Regression: 'thanks' after several off-topic turns must reach the
    farewell branch — the threshold check no longer pre-empts it."""
    scripted_llm.queue({"is_farewell": True, "message": "Glad I could help."})
    state = {
        "messages": [HumanMessage(content="thanks")],
        "clarification_attempts": nodes.MAX_CLARIFICATION_ATTEMPTS + 3,  # well past
    }
    out = nodes.ask_clarification_unknown(state)
    assert out["clarification_attempts"] == 0
    assert "escalated" not in out
    assert "Glad" in out["pending_response"]


def test_ask_clarification_unknown_escalates_when_not_farewell_and_at_threshold(scripted_llm):
    scripted_llm.queue({"is_farewell": False, "message": "(ignored)"})
    state = {
        "messages": [HumanMessage(content="what's the capital of France?")],
        "clarification_attempts": nodes.MAX_CLARIFICATION_ATTEMPTS,
    }
    out = nodes.ask_clarification_unknown(state)
    assert out.get("escalated") is True
    assert out.get("escalation_reason") == "max_clarification_attempts"
    assert nodes.HELP_URL in out["pending_response"]


def test_ask_clarification_unknown_clarifies_below_threshold(scripted_llm):
    scripted_llm.queue({
        "is_farewell": False,
        "message": "I can help with order status, returns, refunds, and policy.",
    })
    state = {
        "messages": [HumanMessage(content="??")],
        "clarification_attempts": 1,
    }
    out = nodes.ask_clarification_unknown(state)
    assert "escalated" not in out
    assert "clarification_attempts" not in out  # not reset, not incremented here
    assert "order status" in out["pending_response"]
