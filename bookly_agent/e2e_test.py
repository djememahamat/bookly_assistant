"""End-to-end scenario runner. Exercises each terminal branch of the graph."""
from langchain_core.messages import HumanMessage
from agent import app

SCENARIOS = [
    (
        "refund_happy_path",
        "want to get a refund, books arrived damaged for order 1043, postcode M13 9PL",
        {"intent": "return_refund", "expect_status": "ok"},
    ),
    (
        "refund_outside_window",
        "I want a refund for order 1045, postcode BS1 6DW, the book was damaged",
        {"intent": "return_refund", "expect_terminal": "respond_ineligible"},
    ),
    (
        "refund_identity_failed",
        "refund for order 1043, postcode XX99 9XX, books were damaged",
        {"intent": "return_refund", "expect_terminal": "respond_identity_failed"},
    ),
    (
        "order_status",
        "where is my order 1042?",
        {"intent": "order_status", "expect_status": "ok"},
    ),
    (
        "general_question",
        "how long does UK delivery take?",
        {"intent": "general_question"},
    ),
    (
        "unknown_intent",
        "what's the weather in Oxford today?",
        {"intent": "unknown"},
    ),
]


def run_one(name: str, prompt: str, expect: dict) -> bool:
    print(f"\n=== {name} ===")
    print(f"USER: {prompt}")
    res = app.invoke({"messages": [HumanMessage(content=prompt)]})
    reply = res["messages"][-1].content.strip()
    intent = res.get("intent")
    tool_results = res.get("tool_results", [])
    print(f"INTENT: {intent}")
    print(f"TOOLS: {[t.get('tool') for t in tool_results]}")
    print(f"AGENT: {reply}")

    ok = True
    if "intent" in expect and intent != expect["intent"]:
        print(f"  FAIL: expected intent={expect['intent']!r}, got {intent!r}")
        ok = False
    if "expect_status" in expect:
        statuses = [t.get("status") for t in tool_results]
        if expect["expect_status"] not in statuses:
            print(f"  FAIL: expected a tool with status={expect['expect_status']!r}, got {statuses}")
            ok = False
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    results = [(name, run_one(name, prompt, expect)) for name, prompt, expect in SCENARIOS]
    print("\n=== SUMMARY ===")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    failed = sum(1 for _, ok in results if not ok)
    print(f"\n{len(results) - failed}/{len(results)} passed")
    raise SystemExit(0 if failed == 0 else 1)
