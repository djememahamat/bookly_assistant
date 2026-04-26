"""L4 — End-to-end tests against the compiled graph.

Per the LangGraph testing guide, we drive the graph via app.invoke(...) and
assert on the final state. The scripted LLM fixture queues exactly the LLM
responses each path will request, so these tests are deterministic, fast,
and free.

Each test below exercises one terminal path of the graph:

  START -> classify_intent -> {gather_arguments, answer_general, ask_clarification}
        -> {ask_clarification, verify_identity, execute_action}
        -> {check_eligibility, respond_identity_failed}
        -> {execute_action, respond_ineligible}
        -> respond -> END

LLM call sequence per path (this is the "script" each test must queue):

  refund_happy:           [classify, gather, respond]                  3 LLM calls
  refund_outside_window:  [classify, gather]                           2 LLM calls
  refund_not_yet_delivered: [classify, gather]                         2 LLM calls
  refund_identity_failed: [classify, gather]                           2 LLM calls
  refund_missing_args:    [classify, gather, ask_clarification]        3 LLM calls
  order_status_happy:     [classify, gather, respond]                  3 LLM calls
  order_status_not_found: [classify, gather, respond]                  3 LLM calls
  general_kb_hit:         [classify, picker, composer]                 3 LLM calls
  general_no_topic:       [classify, picker]                           2 LLM calls
  unknown_intent:         [classify, ask_clarification]                2 LLM calls
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage


# ============================================================
# Happy paths
# ============================================================

def test_e2e_refund_happy_path(app, scripted_llm):
    scripted_llm.queue(
        {"intent": "return_refund"},
        {"order_id": "1043", "postcode": "M13 9PL", "reason": "damaged"},
        "Refund of £14.99 issued for order 1043. Funds will arrive within 3-5 business days.",
    )

    res = app.invoke({"messages": [HumanMessage(content="refund order 1043 postcode M13 9PL damaged")]})

    assert res["intent"] == "return_refund"
    assert res["identity_verified"] is True
    assert res["eligibility_result"]["eligible"] is True
    assert res["eligibility_result"]["refund_amount"] == 22.98 or res["eligibility_result"]["refund_amount"] == 14.99
    tool_names = [t["tool"] for t in res["tool_results"]]
    assert tool_names == ["verify_identity", "check_return_eligibility", "issue_refund"]
    assert res["tool_results"][-1]["result"]["success"] is True
    assert res["tool_results"][-1]["result"]["currency"] == "GBP"
    assert "£14.99" in res["messages"][-1].content


def test_e2e_order_status_happy(app, scripted_llm):
    scripted_llm.queue(
        {"intent": "order_status"},
        {"order_id": "1042"},
        "Order 1042 is out for delivery via Royal Mail, ETA 2026-04-23.",
    )
    res = app.invoke({"messages": [HumanMessage(content="where is my order 1042?")]})

    assert res["intent"] == "order_status"
    assert res["tool_results"][0]["tool"] == "order_lookup"
    assert res["tool_results"][0]["result"]["found"] is True
    assert res["tool_results"][0]["result"]["status"] == "out_for_delivery"


def test_e2e_general_question_kb_hit(app, scripted_llm):
    scripted_llm.queue(
        {"intent": "general_question"},
        "uk_delivery_time",
        "UK orders typically arrive in 2-4 working days from dispatch.",
    )
    res = app.invoke({"messages": [HumanMessage(content="how long does delivery take?")]})

    assert res["intent"] == "general_question"
    assert res["tool_results"][0]["tool"] == "lookup_policy"
    assert res["tool_results"][0]["args"]["topic"] == "uk_delivery_time"
    assert "2-4 working days" in res["messages"][-1].content


# ============================================================
# Non-happy paths
# ============================================================

def test_e2e_refund_outside_window(app, scripted_llm):
    scripted_llm.queue(
        {"intent": "return_refund"},
        {"order_id": "1045", "postcode": "BS1 6DW", "reason": "damaged"},
    )
    res = app.invoke({"messages": [HumanMessage(content="refund 1045 BS1 6DW damaged")]})

    assert res["identity_verified"] is True
    assert res["eligibility_result"]["eligible"] is False
    assert res["eligibility_result"]["reason"] == "outside_return_window"
    tool_names = [t["tool"] for t in res["tool_results"]]
    assert tool_names == ["verify_identity", "check_return_eligibility"]
    assert "issue_refund" not in tool_names
    assert "30-day return" in res["messages"][-1].content


def test_e2e_refund_not_yet_delivered(app, scripted_llm):
    # Order 1046 is in_transit
    scripted_llm.queue(
        {"intent": "return_refund"},
        {"order_id": "1046", "postcode": "EH6 8LN", "reason": "damaged"},
    )
    res = app.invoke({"messages": [HumanMessage(content="refund 1046 EH6 8LN damaged")]})

    assert res["eligibility_result"]["reason"] == "not_yet_delivered"
    assert "hasn't been delivered" in res["messages"][-1].content


def test_e2e_refund_identity_failed(app, scripted_llm):
    scripted_llm.queue(
        {"intent": "return_refund"},
        {"order_id": "1043", "postcode": "XX99 9XX", "reason": "damaged"},
    )
    res = app.invoke({"messages": [HumanMessage(content="refund 1043 XX99 9XX damaged")]})

    assert res["identity_verified"] is False
    # Eligibility never runs when identity fails
    tool_names = [t["tool"] for t in res["tool_results"]]
    assert tool_names == ["verify_identity"]
    assert "couldn't verify your identity" in res["messages"][-1].content


def test_e2e_refund_missing_arguments_routes_to_clarification(app, scripted_llm):
    scripted_llm.queue(
        {"intent": "return_refund"},
        {"order_id": "1043"},  # only order_id; postcode + reason missing
        "What's the postcode on the order, and what's wrong with it?",
    )
    res = app.invoke({"messages": [HumanMessage(content="refund 1043")]})

    assert set(res["missing_arguments"]) == {"postcode", "reason"}
    assert res["awaiting_user_input"] is True
    assert "postcode" in res["messages"][-1].content.lower()


def test_e2e_order_status_not_found(app, scripted_llm):
    scripted_llm.queue(
        {"intent": "order_status"},
        {"order_id": "9999"},
        "I couldn't find order 9999 — could you double-check the number?",
    )
    res = app.invoke({"messages": [HumanMessage(content="where is order 9999?")]})

    assert res["tool_results"][0]["result"]["found"] is False
    # Status is still "ok" — the lookup succeeded, the order just doesn't exist.
    assert res["tool_results"][0]["status"] == "ok"


def test_e2e_general_question_no_kb_topic_match(app, scripted_llm):
    scripted_llm.queue(
        {"intent": "general_question"},
        "none",  # picker says no topic matches
    )
    res = app.invoke({"messages": [HumanMessage(content="why is the sky blue?")]})

    # When picker returns "none", answer_general doesn't call lookup_policy
    # and doesn't make a second LLM call — it returns the help-URL fallback.
    assert "bookly.com/help" in res["messages"][-1].content


def test_e2e_unknown_intent_asks_clarification(app, scripted_llm):
    scripted_llm.queue(
        {"intent": "unknown"},
        {"is_farewell": False, "message": "I help with Bookly orders — what can I do for you?"},
    )
    res = app.invoke({"messages": [HumanMessage(content="what's the weather?")]})

    assert res["intent"] == "unknown"
    assert res["clarification_attempts"] == 1
    assert res["awaiting_user_input"] is True


# ============================================================
# Edge cases
# ============================================================

def test_e2e_refund_change_of_mind_refunds_subtotal_only(app, scripted_llm):
    """Damage keyword absent -> refund_amount = subtotal (no postage)."""
    scripted_llm.queue(
        {"intent": "return_refund"},
        # Use 1048 which has shipping=2.99 and total=22.98 vs subtotal=19.99
        {"order_id": "1048", "postcode": "M3 3NW", "reason": "changed my mind"},
        "Refund of £19.99 issued for order 1048.",
    )
    res = app.invoke({"messages": [HumanMessage(content="changed my mind on 1048, M3 3NW")]})

    assert res["eligibility_result"]["damage_claim"] is False
    assert res["eligibility_result"]["refund_amount"] == 19.99  # subtotal only
    assert res["tool_results"][-1]["result"]["amount"] == 19.99


def test_e2e_refund_postcode_case_and_whitespace_still_verifies(app, scripted_llm):
    """Identity verification is case-insensitive and whitespace-tolerant."""
    scripted_llm.queue(
        {"intent": "return_refund"},
        {"order_id": "1043", "postcode": "  m13 9pl  ", "reason": "damaged"},
        "Refund issued.",
    )
    # Message must literally contain order_id + postcode for the
    # anti-hallucination guard in gather_arguments to keep them.
    res = app.invoke({"messages": [HumanMessage(content="refund 1043 m13 9pl damaged")]})

    assert res["identity_verified"] is True
    assert res["tool_results"][-1]["result"]["success"] is True


def test_e2e_refund_drops_llm_completed_partial_postcode(app, scripted_llm):
    """If the LLM auto-completes a partial postcode the user typed
    (e.g. 'OX4 1H' -> 'OX4 1HB'), gather_arguments must drop the
    extracted value so identity verification can't be bypassed."""
    scripted_llm.queue(
        {"intent": "return_refund"},
        # Gemini "helpfully" completes OX4 1H -> OX4 1HB (the real postcode
        # for order 1044). With the guard, this gets discarded.
        {"order_id": "1044", "postcode": "OX4 1HB", "reason": "missing pages"},
        "What's the postcode on the order?",
    )
    res = app.invoke({"messages": [HumanMessage(content="refund 1044 postcode OX4 1H missing pages")]})

    # postcode was dropped, so we should NOT have proceeded to verification
    assert "postcode" in res["missing_arguments"]
    assert res["awaiting_user_input"] is True
    # No identity check ran, no refund issued
    tool_names = [t["tool"] for t in res.get("tool_results", [])]
    assert "verify_identity" not in tool_names
    assert "issue_refund" not in tool_names


def test_e2e_tool_results_accumulate_via_reducer(app, scripted_llm):
    """The operator.add reducer on tool_results should yield 3 entries
    on the refund happy path (verify_identity, check_eligibility, issue_refund)."""
    scripted_llm.queue(
        {"intent": "return_refund"},
        {"order_id": "1043", "postcode": "M13 9PL", "reason": "damaged"},
        "Refund issued.",
    )
    res = app.invoke({"messages": [HumanMessage(content="refund 1043 M13 9PL damaged")]})
    assert len(res["tool_results"]) == 3


def test_e2e_messages_reducer_appends_user_then_agent(app, scripted_llm):
    """add_messages reducer: input HumanMessage + agent's AIMessage = 2 messages."""
    scripted_llm.queue(
        {"intent": "general_question"},
        "uk_delivery_time",
        "2-4 working days.",
    )
    res = app.invoke({"messages": [HumanMessage(content="delivery time?")]})
    assert len(res["messages"]) == 2
    assert res["messages"][0].content == "delivery time?"
    assert res["messages"][1].content == "2-4 working days."
