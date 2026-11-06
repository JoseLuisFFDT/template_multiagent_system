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

## Start the system

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/) and a [Groq](https://console.groq.com) API key.

**One-time setup** (each project has its own virtual environment):

```powershell
cd backend
uv sync                          # creates backend/.venv and installs its dependencies
copy .env.example .env           # then edit .env and set GROQ_API_KEY
cd ..\frontend
uv sync                          # creates frontend/.venv
```

**Every time**, in **two terminals** (the backend must be running before you use the chat):

| | Terminal 1: backend | Terminal 2: frontend |
|---|---|---|
| Command | `cd backend` then `uv run python main.py` | `cd frontend` then `uv run streamlit run app.py` |
| Listens on | http://127.0.0.1:8000 (interactive docs at `/docs`) | http://localhost:8501 (opens the browser) |
| Ready when | the terminal prints `Uvicorn running on http://127.0.0.1:8000` | the browser shows the chat |
| Stop it | `Ctrl+C` in that terminal | `Ctrl+C` in that terminal |

Always launch with `uv run` from inside each folder: it picks that folder's own environment, even if another one is active.
In VS Code you can also use *Run and Debug* (Ctrl+Shift+D) and choose **Backend: run API (main.py)** or **Frontend: Streamlit chat**.

If the chat says `Could not reach the backend`, the backend is not running (or is on another port: set `PORT` for the backend
and `API_URL` for the frontend). If `streamlit` is "not installed", you are using the wrong environment: run it with `uv run`
from `frontend/`.

## Talk to the system from the terminal

You do not need the frontend to use the system. Pick one:

| Way | Needs the backend running? | Command (from `backend/` unless noted) |
|---|---|---|
| **Interactive chat in the terminal** | No | `uv run python cli.py --chat`. Type a request, read the answer; an empty line exits. It prints each node as it runs. |
| **One request in the terminal** | No | `uv run python cli.py "Reverse the text 'langgraph'"` |
| **HTTP request (PowerShell)** | Yes | see below |
| **Interactive API docs** | Yes | open http://127.0.0.1:8000/docs, expand `POST /chat`, *Try it out* |

Sending a prompt to the running backend from another terminal:

```powershell
$body  = @{ message = "Reverse the text 'langgraph'" } | ConvertTo-Json
$bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
$r = Invoke-RestMethod -Uri http://127.0.0.1:8000/chat -Method Post `
       -ContentType "application/json; charset=utf-8" -Body $bytes
$r.route; $r.answer               # which agent answered, and the answer
```

If `$r.status` is `needs_approval` the graph is paused waiting for a human; answer it with
`Invoke-RestMethod -Uri http://127.0.0.1:8000/approve -Method Post -ContentType "application/json" -Body (@{ thread_id = $r.thread_id; approved = $true } | ConvertTo-Json)`.
More examples (including `curl.exe`) and the full response format are in [backend/README.md](backend/README.md#call-the-api-from-the-terminal).

Want to read the multi-agent code first? Start with [backend/README.md](backend/README.md): it explains the graph, the states, the
subgraph and how to add your own agent.

```text
.
├── backend/     # src/, tests/, main.py, cli.py, agents_config.yaml, pyproject.toml, uv.lock, .env
├── frontend/    # app.py, pyproject.toml, uv.lock
└── .vscode/     # launch configurations for both projects
```
