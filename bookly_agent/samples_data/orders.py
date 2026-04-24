from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Address:
    line1: str
    line2: str
    city: str
    postcode: str
    country: str


@dataclass
class Item:
    sku: str
    title: str
    author: str
    price: float
    quantity: int


@dataclass
class Order:
    order_id: str
    customer_email: str
    customer_name: str
    items: list[Item]
    subtotal: float
    shipping: float
    currency: str
    total: float
    status: str
    placed_at: str
    shipping_address: Address
    shipped_at: Optional[str] = None
    delivered_at: Optional[str] = None
    estimated_delivery: Optional[str] = None
    carrier: Optional[str] = None
    service_level: Optional[str] = None
    tracking_number: Optional[str] = None


def _parse(data: dict) -> Order:
    return Order(
        **{k: v for k, v in data.items() if k not in ("items", "shipping_address")},
        items=[Item(**i) for i in data["items"]],
        shipping_address=Address(**data["shipping_address"]),
    )


def load_orders(path: Path | str | None = None) -> dict[str, Order]:
    """Load all orders from the JSON file, keyed by order_id."""
    if path is None:
        path = Path(__file__).parent.parent / "samples_data" / "orders.json"
    raw = json.loads(Path(path).read_text())
    return {oid: _parse(data) for oid, data in raw.items()}


# Module-level singleton so tools can just import ORDERS.
ORDERS: dict[str, Order] = load_orders()