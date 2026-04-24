import uuid
from dataclasses import asdict
from datetime import datetime, timezone

from langchain.tools import tool
from samples_data.orders import ORDERS
from samples_data.kb import POLICIES, AVAILABLE_POLICIES_TOPICS

RETURN_WINDOW_DAYS = 30

@tool
def order_lookup(order_id: str) -> dict:
    """
    Check whether a given order exists.

    params order_id: order_id to retrieve
    returns: {"found": True, ...order} if it exists, otherwise {"found": False, "order_id": order_id}.
    """
    order = ORDERS.get(str(order_id))
    if not order:
        return {"found": False, "order_id": order_id}
    return {"found": True, **asdict(order)}

@tool
def lookup_policy(topic: str) -> dict:
    """Return Bookly's policy text for a named topic.

    Data source: samples_data/kb.py, which parses samples_data/policies.md at import time.
    This means editing the canonical policy markdown updates the agent's
    knowledge with no code change.
    """
    if topic not in POLICIES:
        return {
            "found": False,
            "requested_topic": topic,
            "available_topics": AVAILABLE_POLICIES_TOPICS,
        }
    return {"found": True, "topic": topic, "content": POLICIES[topic]}

@tool
def verify_identity(order_id: str, claimed_postcode: str) -> dict:
    """Verify that claimed_postcode matches the postcode on database for order_id.
    """
    order = ORDERS.get(str(order_id).strip())
    if not order:
        return {"verified": False, "reason": "order_not_found"}

    match = order.shipping_address.postcode.lower().strip() == claimed_postcode.lower().strip()
    return {
        "verified": match,
        "reason": "postcode_matches" if match else "postcode_mismatch",
    }

@tool
def check_return_eligibility(order_id: str, reason: str) -> dict:
    """Evaluate whether an order is eligible for return under Bookly policy.

    This is pure policy code. It returns structured facts — eligibility,
    amount, and a machine-readable reason — and never makes value
    judgments. The DOWNSTREAM graph decides what to do with the result
    (auto-approve under cap, escalate over cap, deny if ineligible).
    """
    order = ORDERS.get(str(order_id).strip())
    if not order:
        return {"eligible": False, "reason": "order_not_found", "refund_amount": 0.0}

    if order.status != "delivered":
        return {
            "eligible": False,
            "reason": "not_yet_delivered",
            "refund_amount": 0.0,
            "current_status": order.status,
        }

    delivered_at = datetime.fromisoformat(order.delivered_at).replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    days_since_delivery = (now - delivered_at).days

    if days_since_delivery > RETURN_WINDOW_DAYS:
        return {
            "eligible": False,
            "reason": "outside_return_window",
            "refund_amount": 0.0,
            "days_since_delivery": days_since_delivery,
            "window_days": RETURN_WINDOW_DAYS,
        }

    is_damaged = any(k in reason.lower() for k in ["damag", "defect", "broken", "wrong item"])
    refund_amount = order.total if is_damaged else order.subtotal

    return {
        "eligible": True,
        "reason": "within_policy",
        "refund_amount": round(refund_amount, 2),
        "days_since_delivery": days_since_delivery,
        "damage_claim": is_damaged,
    }

@tool
def issue_refund(order_id: str, amount: float, identity_verified: bool) -> dict:
    """Issue a refund. Refuses if identity was not verified.
 
    Identity verification is enforced in code as a final safety check,
    in addition to being a topological precondition in the graph.
    Refund amount is always computed from actual order data upstream
    in check_return_eligibility — never from user input — so no
    amount-based ceiling is enforced here.
    """
    if not identity_verified:
        return {
            "success": False,
            "reason": "identity_not_verified",
            "message": "Identity verification is required before issuing refunds.",
        }
    if amount <= 0:
        return {"success": False, "reason": "invalid_amount"}

    order = ORDERS.get(str(order_id).strip())
    if not order:
        return {"success": False, "reason": "order_not_found"}

    # In production: call Stripe refund API, write to orders DB, emit event.
    refund_id = f"RF-{uuid.uuid4().hex[:8].upper()}"
    return {
        "success": True,
        "refund_id": refund_id,
        "order_id": order_id,
        "amount": round(amount, 2),
        "currency": order.currency,
        "timing": "3-5 business days",
    }

#Tools exposed to the LLM
LLM_TOOLS = [order_lookup, lookup_policy, verify_identity, check_return_eligibility]