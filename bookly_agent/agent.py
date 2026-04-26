from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, START, END

from state import BooklyAgentState
import nodes
# from nodes import (
#     # run_agent_reasoning,
#     # tool_node
#     classify_intent
# )

# AGENT_REASON="agent_reason"
# ACT="act"
# LAST=-1

# def should_continue(state: MessagesState) -> str:
#     if not state["messages"][LAST].tool_calls:
#         return END
#     return ACT

# flow = StateGraph(MessagesState)

# flow.add_node(AGENT_REASON, run_agent_reasoning)
# flow.add_node(ACT, tool_node)


# flow.add_edge(START, AGENT_REASON)
# flow.add_conditional_edges(AGENT_REASON, should_continue, {
#     END:END,
#     ACT:ACT})
# flow.add_edge(ACT, AGENT_REASON)
CLASSIFY_INTENT = "classify_intent"
GATHER_ARGUMENTS = "gather_arguments"
ASK_CLARIFICATION = "ask_clarification"
EXECUTE_ACTION = "execute_action"
RESPOND = "respond"
ANSWER_GENERAL = "answer_general"
ASK_CLARIFICATION_UNKNOWN = "ask_clarification_unknown"
VERIFY_IDENTITY = "verify_identity"
RESPOND_IDENTITY_FAILED = "respond_identity_failed"
CHECK_ELIGIBILITY = "check_eligibility"
RESPOND_INELIGIBLE = "respond_ineligible"

#Build the graph (state machine)
builder = StateGraph(BooklyAgentState)

#Add nodes
builder.add_node(CLASSIFY_INTENT, nodes.classify_intent)
builder.add_node(GATHER_ARGUMENTS, nodes.gather_arguments)
builder.add_node(ASK_CLARIFICATION, nodes.ask_clarification)
builder.add_node(ASK_CLARIFICATION_UNKNOWN, nodes.ask_clarification_unknown)
builder.add_node(EXECUTE_ACTION, nodes.execute_action)
builder.add_node(RESPOND, nodes.respond)
builder.add_node(ANSWER_GENERAL, nodes.answer_general)
builder.add_node(VERIFY_IDENTITY, nodes.verify_identity)
builder.add_node(RESPOND_IDENTITY_FAILED, nodes.respond_identity_failed)
builder.add_node(CHECK_ELIGIBILITY, nodes.check_eligibility)
builder.add_node(RESPOND_INELIGIBLE, nodes.respond_ineligible)

#Add Edges
builder.add_edge(START, CLASSIFY_INTENT)
builder.add_conditional_edges(CLASSIFY_INTENT, nodes.route_after_classify, {
    GATHER_ARGUMENTS: GATHER_ARGUMENTS,
    ANSWER_GENERAL: ANSWER_GENERAL,
    ASK_CLARIFICATION_UNKNOWN: ASK_CLARIFICATION_UNKNOWN,
})
builder.add_conditional_edges(GATHER_ARGUMENTS, nodes.route_after_gather_arguments,{
    ASK_CLARIFICATION: ASK_CLARIFICATION,
    EXECUTE_ACTION: EXECUTE_ACTION,
    VERIFY_IDENTITY: VERIFY_IDENTITY
})
builder.add_conditional_edges(VERIFY_IDENTITY, nodes.route_after_verify_identity,{
    CHECK_ELIGIBILITY: CHECK_ELIGIBILITY,
    RESPOND_IDENTITY_FAILED: RESPOND_IDENTITY_FAILED
})
builder.add_conditional_edges(CHECK_ELIGIBILITY, nodes.route_after_eligibility,{
    EXECUTE_ACTION: EXECUTE_ACTION,         # eligible
    RESPOND_INELIGIBLE: RESPOND_INELIGIBLE # not eligible
})

builder.add_edge(EXECUTE_ACTION, RESPOND)

# --- Terminal nodes all yield control to the outer loop.
builder.add_edge(ASK_CLARIFICATION, END)
builder.add_edge(ASK_CLARIFICATION_UNKNOWN, END)
builder.add_edge(RESPOND, END)
builder.add_edge(ANSWER_GENERAL, END)
builder.add_edge(RESPOND_IDENTITY_FAILED, END)
builder.add_edge(RESPOND_INELIGIBLE, END)

app = builder.compile()

if __name__ == "__main__":
    # app.get_graph().draw_mermaid_png(output_file_path="flow.png")
    res = app.invoke({"messages": [HumanMessage(content="want to get a refund, the books received were damaged for the order 1043, here my postcode M13 9PL ")]})
    print(res["messages"][-1].content)





