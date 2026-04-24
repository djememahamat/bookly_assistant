"""L1 — Tool unit tests. Pure functions, no LLM.

Asserts every branch of every @tool in tools.py."""
from __future__ import annotations

import pytest

import tools as bookly_tools


# ---------- order_lookup ----------

def test_order_lookup_found():
    out = bookly_tools.order_lookup.invoke({"order_id": "1043"})
    assert out["found"] is True
    assert out["order_id"] == "1043"
    assert out["customer_name"] == "Maya Patel"


def test_order_lookup_not_found():
    out = bookly_tools.order_lookup.invoke({"order_id": "9999"})
    assert out == {"found": False, "order_id": "9999"}


# ---------- verify_identity ----------

def test_verify_identity_match():
    out = bookly_tools.verify_identity.invoke({"order_id": "1043", "claimed_postcode": "M13 9PL"})
    assert out == {"verified": True, "reason": "postcode_matches"}


def test_verify_identity_mismatch():
    out = bookly_tools.verify_identity.invoke({"order_id": "1043", "claimed_postcode": "XX99 9XX"})
    assert out == {"verified": False, "reason": "postcode_mismatch"}


def test_verify_identity_order_not_found():
    out = bookly_tools.verify_identity.invoke({"order_id": "9999", "claimed_postcode": "M13 9PL"})
    assert out == {"verified": False, "reason": "order_not_found"}


@pytest.mark.parametrize("claimed", ["m13 9pl", "  M13 9PL  ", "M13   9PL".replace("   ", " ")])
def test_verify_identity_case_and_whitespace(claimed):
    out = bookly_tools.verify_identity.invoke({"order_id": "1043", "claimed_postcode": claimed})
    assert out["verified"] is True


# ---------- check_return_eligibility ----------

def test_eligibility_within_window_change_of_mind_refunds_subtotal_only():
    # 1043 delivered 2026-04-08, today is 2026-04-24 -> 16 days, within window.
    out = bookly_tools.check_return_eligibility.invoke({"order_id": "1043", "reason": "changed my mind"})
    assert out["eligible"] is True
    assert out["damage_claim"] is False
    assert out["refund_amount"] == 14.99  # subtotal, no shipping (it was 0 anyway)


def test_eligibility_within_window_damage_refunds_total_with_shipping():
    # 1048 delivered 2026-04-06, total 22.98 (subtotal 19.99 + shipping 2.99).
    out = bookly_tools.check_return_eligibility.invoke({"order_id": "1048", "reason": "damaged on arrival"})
    assert out["eligible"] is True
    assert out["damage_claim"] is True
    assert out["refund_amount"] == 22.98


@pytest.mark.parametrize("reason", ["damaged", "damage", "defective binding", "broken spine", "wrong item arrived"])
def test_eligibility_damage_keyword_variants(reason):
    out = bookly_tools.check_return_eligibility.invoke({"order_id": "1048", "reason": reason})
    assert out["damage_claim"] is True


def test_eligibility_outside_return_window():
    # 1045 delivered 2026-02-21, well outside 30 days as of 2026-04-24.
    out = bookly_tools.check_return_eligibility.invoke({"order_id": "1045", "reason": "damaged"})
    assert out["eligible"] is False
    assert out["reason"] == "outside_return_window"
    assert out["window_days"] == 30
    assert out["days_since_delivery"] > 30


def test_eligibility_not_yet_delivered():
    # 1042 is out_for_delivery
    out = bookly_tools.check_return_eligibility.invoke({"order_id": "1042", "reason": "damaged"})
    assert out["eligible"] is False
    assert out["reason"] == "not_yet_delivered"
    assert out["current_status"] == "out_for_delivery"


def test_eligibility_processing_status():
    # 1047 is processing — also "not yet delivered"
    out = bookly_tools.check_return_eligibility.invoke({"order_id": "1047", "reason": "damaged"})
    assert out["eligible"] is False
    assert out["reason"] == "not_yet_delivered"


def test_eligibility_in_transit_status():
    # 1046 is in_transit
    out = bookly_tools.check_return_eligibility.invoke({"order_id": "1046", "reason": "damaged"})
    assert out["eligible"] is False
    assert out["reason"] == "not_yet_delivered"


def test_eligibility_order_not_found():
    out = bookly_tools.check_return_eligibility.invoke({"order_id": "9999", "reason": "damaged"})
    assert out["eligible"] is False
    assert out["reason"] == "order_not_found"
    assert out["refund_amount"] == 0.0


# ---------- issue_refund ----------

def test_issue_refund_happy_path():
    out = bookly_tools.issue_refund.invoke({"order_id": "1043", "amount": 14.99, "identity_verified": True})
    assert out["success"] is True
    assert out["refund_id"].startswith("RF-")
    assert out["amount"] == 14.99
    assert out["currency"] == "GBP"
    assert out["timing"] == "3-5 business days"


def test_issue_refund_blocks_when_identity_unverified():
    out = bookly_tools.issue_refund.invoke({"order_id": "1043", "amount": 14.99, "identity_verified": False})
    assert out["success"] is False
    assert out["reason"] == "identity_not_verified"


@pytest.mark.parametrize("amount", [0.0, -1.0, -0.01])
def test_issue_refund_rejects_invalid_amount(amount):
    out = bookly_tools.issue_refund.invoke({"order_id": "1043", "amount": amount, "identity_verified": True})
    assert out["success"] is False
    assert out["reason"] == "invalid_amount"


def test_issue_refund_order_not_found():
    out = bookly_tools.issue_refund.invoke({"order_id": "9999", "amount": 10.0, "identity_verified": True})
    assert out["success"] is False
    assert out["reason"] == "order_not_found"


# ---------- lookup_policy ----------

def test_lookup_policy_found():
    out = bookly_tools.lookup_policy.invoke({"topic": "return_window"})
    assert out["found"] is True
    assert out["topic"] == "return_window"
    assert "30 days" in out["content"]


def test_lookup_policy_unknown_topic_lists_alternatives():
    out = bookly_tools.lookup_policy.invoke({"topic": "made_up_topic"})
    assert out["found"] is False
    assert out["requested_topic"] == "made_up_topic"
    assert isinstance(out["available_topics"], list)
    assert "return_window" in out["available_topics"]
