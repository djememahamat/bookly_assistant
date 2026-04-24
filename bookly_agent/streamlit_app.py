"""Streamlit chat UI for the Bookly support agent.

Demo-focused: clean transcript, no graph internals. Same multi-turn contract
as the CLI — MemorySaver-backed persistence keyed by a per-session thread id.

Run:
    uv run streamlit run bookly_agent/streamlit_app.py
"""
from __future__ import annotations
import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from agent import builder
from safety import sanitize_user_input

ASSETS = Path(__file__).parent / "assets"
BOOKS_ICON = ASSETS / "books.png"
USER_ICON = ASSETS / "user.png"

PAGE_ICON = str(BOOKS_ICON) if BOOKS_ICON.exists() else "📚"
ASSISTANT_AVATAR = str(BOOKS_ICON) if BOOKS_ICON.exists() else None
USER_AVATAR = str(USER_ICON) if USER_ICON.exists() else None


@st.cache_resource(show_spinner=False)
def get_app():
    # Compiled once per Streamlit process. MemorySaver isolates browser
    # sessions internally via thread_id, so one app serves every visitor.
    return builder.compile(checkpointer=MemorySaver())


def init_session() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid.uuid4().hex[:8]
    if "history" not in st.session_state:
        st.session_state.history = []


def reset_conversation() -> None:
    st.session_state.thread_id = uuid.uuid4().hex[:8]
    st.session_state.history = []


st.set_page_config(page_title="Bookly Support", page_icon=PAGE_ICON, layout="centered")

init_session()
app = get_app()

with st.sidebar:
    st.markdown("## Bookly Support")
    st.caption("Order status · Returns & refunds · Policy questions")
    st.button(
        "New conversation",
        on_click=reset_conversation,
        use_container_width=True,
    )

st.title("How can I help?")
st.caption("Ask about an order, start a return, or check our policies.")

def avatar_for(role: str) -> str | None:
    return USER_AVATAR if role == "user" else ASSISTANT_AVATAR


for role, content in st.session_state.history:
    with st.chat_message(role, avatar=avatar_for(role)):
        st.markdown(content)

prompt = st.chat_input("Type your message...")
if prompt:
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    safe = sanitize_user_input(prompt)
    if not safe:
        reply = "Sorry, I didn't catch that — could you rephrase?"
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            st.markdown(reply)
    else:
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            with st.spinner("Thinking..."):
                try:
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    result = app.invoke(
                        {"messages": [HumanMessage(content=safe)]},
                        config=config,
                    )
                    reply = result["messages"][-1].content
                except Exception:
                    reply = "Something went wrong on my side — please try again."
            st.markdown(reply)

    st.session_state.history.append(("assistant", reply))
