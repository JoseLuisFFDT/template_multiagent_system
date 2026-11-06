"""Minimal Streamlit chat for the multi-agent system. It only talks to the HTTP API.

    uv run streamlit run app.py      (from the frontend/ folder)

The backend must be running (``python main.py`` in backend/). Set ``API_URL`` if it is not on localhost:8000.
This file is self-contained: it imports nothing from the backend, it only calls its HTTP API.

Under every answer there is a **graph trace**: each node that ran, what it wrote, and the full state of
its graph right after it (the subgraph steps are marked "inside subgraph_agent").
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Multi-agent chat")
st.title("Multi-agent chat")

# Streamlit re-runs this whole script on every interaction; session_state is what survives.
if "messages" not in st.session_state:
    # [{"role": "user" | "assistant", "content": str, "route": str | None, "trace": list}]
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None  # response of /chat while the graph waits for human approval
if "error" not in st.session_state:
    st.session_state.error = None  # kept in session_state because st.rerun() would erase a plain st.error()


def post(path: str, payload: dict) -> dict | None:
    """Call the API. On failure, remember why (shown by the page) and return None."""
    try:
        response = requests.post(f"{API_URL}{path}", json=payload, timeout=120)
    except requests.RequestException as exc:
        st.session_state.error = f"Could not reach the backend at {API_URL}. Is it running? ({type(exc).__name__})"
        return None
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        st.session_state.error = f"The backend answered {response.status_code}: {detail}"
        return None
    st.session_state.error = None
    return response.json()


def handle(data: dict | None, earlier_trace: list | None = None) -> None:
    """Store an API response: a finished answer goes to the history, a pause becomes `pending`."""
    if data is None:
        return
    trace = (earlier_trace or []) + data["trace"]
    for number, step in enumerate(trace, start=1):  # /approve numbers its steps from 1 again
        step["index"] = number
    if data["status"] == "completed":
        st.session_state.pending = None
        st.session_state.messages.append(
            {"role": "assistant", "content": data["answer"], "route": data["route"], "trace": trace}
        )
    else:
        st.session_state.pending = {**data, "trace": trace}


def render_trace(trace: list) -> None:
    """Show how the graph state evolved: one block per executed node."""
    if not trace:
        return
    with st.expander(f"Graph trace ({len(trace)} steps)"):
        for step in trace:
            inside = "" if step["graph"] == "main" else f"  (inside {step['graph']})"
            st.markdown(f"**{step['index']}. `{step['node']}`**{inside}")
            wrote, state = st.columns(2)
            wrote.caption("wrote (what the node returned)")
            wrote.json(step["update"] if step["update"] is not None else {}, expanded=True)
            state.caption("state after this step")
            state.json(step["state"] if step["state"] is not None else {}, expanded=False)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("route"):
            st.caption(f"answered by {message['route']}")
        render_trace(message.get("trace", []))

pending = st.session_state.pending
if pending:
    with st.chat_message("assistant"):
        st.markdown(f"**{pending['question']}**")
        st.markdown(pending["preview"])
        render_trace(pending["trace"])
        approve_col, reject_col = st.columns(2)
        decision = True if approve_col.button("Approve") else False if reject_col.button("Reject") else None
    if decision is not None:
        handle(post("/approve", {"thread_id": pending["thread_id"], "approved": decision}), pending["trace"])
        st.rerun()

if st.session_state.error:
    st.error(st.session_state.error)

# The input is disabled while a decision is pending: the graph is paused on that run.
if prompt := st.chat_input("Write a request for the mastermind", disabled=pending is not None):
    st.session_state.messages.append({"role": "user", "content": prompt, "route": None, "trace": []})
    with st.spinner("Thinking..."):
        handle(post("/chat", {"message": prompt}))
    st.rerun()
