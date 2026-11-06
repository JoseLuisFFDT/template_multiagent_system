# template_multiagent_system

A deliberately small **multi-agent system template**, split into two fully independent projects that talk over HTTP:

```text
 frontend/ (Streamlit chat)  --HTTP-->  backend/ (FastAPI + LangGraph: mastermind -> agents)
```

| Folder | What it is | Docs |
|---|---|---|
| [`backend/`](backend/README.md) | The multi-agent system (LangGraph) behind a FastAPI API. Agents, graphs, tools, MCP, subgraph, tests. | [backend/README.md](backend/README.md) |
| [`frontend/`](frontend/README.md) | A minimal Streamlit chat that only calls the backend API. | [frontend/README.md](frontend/README.md) |

**They are separate on purpose.** Each folder has its own `pyproject.toml`, `uv.lock`, virtual environment and README, and
neither imports code from the other. The only contract between them is the HTTP API of the backend (`POST /chat`,
`POST /approve`) plus the frontend's `API_URL` setting. That is what lets you package and deploy them separately later
(for example as two containers): copy one folder, install its dependencies, run it.

## Run both

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/) and a [Groq](https://console.groq.com) API key.

```powershell
# terminal 1: backend
cd backend
uv sync
copy .env.example .env           # then edit .env and set GROQ_API_KEY
uv run python main.py            # http://127.0.0.1:8000  (docs at /docs)

# terminal 2: frontend
cd frontend
uv sync
uv run streamlit run app.py      # http://localhost:8501
```

Want to read the multi-agent code first? Start with [backend/README.md](backend/README.md): it explains the graph, the states, the
subgraph and how to add your own agent. The backend can also be used without the frontend (`uv run python cli.py --chat`).

```text
.
├── backend/     # src/, tests/, main.py, cli.py, agents_config.yaml, pyproject.toml, uv.lock, .env
├── frontend/    # app.py, pyproject.toml, uv.lock
└── .vscode/     # launch configurations for both projects
```
