# frontend

A minimal **Streamlit chat** for the multi-agent backend: one chat box. Each answer shows which agent produced it. When the
backend pauses for human approval (the `subgraph_agent` case) the UI shows the preview with **Approve / Reject** buttons and
keeps the input disabled until you decide.

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
