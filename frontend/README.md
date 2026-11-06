# frontend

A minimal **Streamlit chat** for the multi-agent backend: one chat box. Each answer shows which agent produced it. When the
backend pauses for human approval (the `subgraph_agent` case) the UI shows the preview with **Approve / Reject** buttons and
keeps the input disabled until you decide.

**Graph trace.** Under every answer there is an expandable *Graph trace* with one block per node that ran: on the left what the
node **wrote** (only the keys it returned), on the right the **full state of its graph after that step**. Steps of the subgraph are
marked "inside subgraph_agent" and show the subgraph's own state. It is the backend's `trace` field, drawn as is.

**Errors.** If the backend is down or fails, the page shows the reason under the chat (for example `Could not reach the backend` or
`The backend answered 502: ...`).

This folder is **self-contained**: its own `pyproject.toml`, `uv.lock` and `.venv`. It does not import any backend code; it only
calls the backend's HTTP API (`POST /chat`, `POST /approve`, documented in `../backend/README.md`).

```powershell
cd frontend
uv sync
uv run streamlit run app.py      # http://localhost:8501  (the backend must be running)
```

| Variable | Default | Meaning |
|---|---|---|
| `API_URL` | `http://localhost:8000` | Where the backend listens. Set it when the backend runs elsewhere (another host or container). |

Everything lives in `app.py`. Streamlit re-runs the script on every interaction; `st.session_state` keeps the chat history and
the pending approval between runs.
