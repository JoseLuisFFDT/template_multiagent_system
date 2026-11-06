"""Minimal Streamlit chat for the multi-agent system. It only talks to the HTTP API.

    streamlit run src/frontend/app.py

The API must be running (``python main.py``). Set ``API_URL`` if it is not on localhost:8000.
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Multi-agent chat")
st.title("Multi-agent chat")

# Streamlit re-runs this whole script on every interaction; session_state is what survives.
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user" | "assistant", "content": str, "route": str | None}]
if "pending" not in st.session_state:
    st.session_state.pending = None  # response of /chat while the graph waits for human approval


def post(path: str, payload: dict) -> dict | None:
    try:
        response = requests.post(f"{API_URL}{path}", json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {API_URL}: {exc}")
        return None


def handle(data: dict | None) -> None:
    """Store an API response: a finished answer goes to the history, a pause becomes `pending`."""
    if data is None:
        return
    if data["status"] == "completed":
        st.session_state.pending = None
        st.session_state.messages.append({"role": "assistant", "content": data["answer"], "route": data["route"]})
    else:
        st.session_state.pending = data


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("route"):
            st.caption(f"answered by {message['route']}")

pending = st.session_state.pending
if pending:
    with st.chat_message("assistant"):
        st.markdown(f"**{pending['question']}**")
        st.markdown(pending["preview"])
        approve_col, reject_col = st.columns(2)
        decision = True if approve_col.button("Approve") else False if reject_col.button("Reject") else None
    if decision is not None:
        handle(post("/approve", {"thread_id": pending["thread_id"], "approved": decision}))
        st.rerun()

# The input is disabled while a decision is pending: the graph is paused on that run.
if prompt := st.chat_input("Write a request for the mastermind", disabled=pending is not None):
    st.session_state.messages.append({"role": "user", "content": prompt, "route": None})
    with st.spinner("Thinking..."):
        handle(post("/chat", {"message": prompt}))
    st.rerun()
