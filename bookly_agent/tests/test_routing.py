"""L3 — Conditional edge / routing unit tests.

Each route_* helper in nodes.py is a pure function over state. Exhaustively
exercise every branch."""
from __future__ import annotations

import pytest

import nodes


# ---------- route_after_classify ----------

@pytest.mark.parametrize(
    "intent,expected",
    [
        ("order_status", "gather_arguments"),
        ("return_refund", "gather_arguments"),
        ("general_question", "answer_general"),
    ],
)
def test_route_after_classify_known_intents(intent, expected):
    assert nodes.route_after_classify({"intent": intent, "clarification_attempts": 0}) == expected


def test_route_after_classify_unknown_below_threshold_asks_clarification():
    state = {"intent": "unknown", "clarification_attempts": 0}
    assert nodes.route_after_classify(state) == "ask_clarification_unknown"


def test_route_after_classify_unknown_at_threshold_escalates():
    """At MAX_CLARIFICATION_ATTEMPTS the helper routes to 'respond_cannot_help',
    which is the terminal node wired in agent.py for this branch."""
    state = {"intent": "unknown", "clarification_attempts": nodes.MAX_CLARIFICATION_ATTEMPTS}
    assert nodes.route_after_classify(state) == "respond_cannot_help"


# ---------- route_after_gather_arguments ----------

def test_route_after_gather_arguments_missing_args_asks_clarification():
    state = {"intent": "return_refund", "missing_arguments": ["postcode"]}
    assert nodes.route_after_gather_arguments(state) == "ask_clarification"


def test_route_after_gather_arguments_return_refund_goes_to_verify_identity():
    state = {"intent": "return_refund", "missing_arguments": []}
    assert nodes.route_after_gather_arguments(state) == "verify_identity"


def test_route_after_gather_arguments_order_status_goes_straight_to_execute():
    state = {"intent": "order_status", "missing_arguments": []}
    assert nodes.route_after_gather_arguments(state) == "execute_action"


def test_route_after_gather_arguments_missing_overrides_intent():
    """Even for return_refund, missing args should not skip clarification."""
    state = {"intent": "return_refund", "missing_arguments": ["order_id"]}
    assert nodes.route_after_gather_arguments(state) == "ask_clarification"


# ---------- route_after_verify_identity ----------

def test_route_after_verify_identity_verified():
    assert nodes.route_after_verify_identity({"identity_verified": True}) == "check_eligibility"


def test_route_after_verify_identity_not_verified():
    assert nodes.route_after_verify_identity({"identity_verified": False}) == "respond_identity_failed"


def test_route_after_verify_identity_missing_flag_treated_as_unverified():
    assert nodes.route_after_verify_identity({}) == "respond_identity_failed"


# ---------- route_after_eligibility ----------

def test_route_after_eligibility_eligible():
    state = {"eligibility_result": {"eligible": True, "refund_amount": 14.99}}
    assert nodes.route_after_eligibility(state) == "execute_action"


def test_route_after_eligibility_not_eligible():
    state = {"eligibility_result": {"eligible": False, "reason": "outside_return_window"}}
    assert nodes.route_after_eligibility(state) == "respond_ineligible"


def test_route_after_eligibility_missing_result_routes_to_ineligible():
    """Defensive: a None or missing eligibility_result should not crash routing."""
    assert nodes.route_after_eligibility({}) == "respond_ineligible"
    assert nodes.route_after_eligibility({"eligibility_result": None}) == "respond_ineligible"
