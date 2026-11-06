"""Tests of the HTTP API with the REAL graph and a stubbed LLM (offline)."""
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from src.agents.chat_agent import ChatAgent
from src.agents.subgraph_agent import SubgraphAgent
from src.api.app import create_app
from src.graphs.main_graph import build_main_graph
from tests.fakes import ScriptedLLM
from tests.test_main_graph import mastermind_routing_to
from tests.test_subgraph import make_subgraph


def chat_client(answer: str = "Hello!") -> TestClient:
    chat = ChatAgent()
    chat.llm = ScriptedLLM([AIMessage(content=answer)])
    graph = build_main_graph({"mastermind_agent": mastermind_routing_to("chat_agent"), "chat_agent": chat})
    return TestClient(create_app(graph))


def subgraph_client() -> TestClient:
    subgraph, _ = make_subgraph()
    graph = build_main_graph(
        {"mastermind_agent": mastermind_routing_to("subgraph_agent"), "subgraph_agent": SubgraphAgent(subgraph=subgraph)}
    )
    return TestClient(create_app(graph))


def test_health():
    assert chat_client().get("/health").json() == {"status": "ok"}


def test_chat_returns_the_final_answer_and_the_route():
    body = chat_client("Hello!").post("/chat", json={"message": "hi"}).json()

    assert body["status"] == "completed"
    assert body["route"] == "chat_agent"
    assert body["answer"] == "Hello!"
    assert body["thread_id"]


def test_empty_message_is_rejected():
    assert chat_client().post("/chat", json={"message": "   "}).status_code == 422


def test_subgraph_request_pauses_then_approve_delivers_the_answer():
    client = subgraph_client()

    paused = client.post("/chat", json={"message": "answer in parallel"}).json()
    assert paused["status"] == "needs_approval"
    assert paused["answer"] is None
    assert "- A: answer 1" in paused["preview"]

    done = client.post("/approve", json={"thread_id": paused["thread_id"], "approved": True}).json()
    assert done["status"] == "completed"
    assert "- A: answer 1" in done["answer"]


def test_reject_returns_the_rejection_message():
    client = subgraph_client()
    paused = client.post("/chat", json={"message": "answer in parallel"}).json()

    done = client.post("/approve", json={"thread_id": paused["thread_id"], "approved": False}).json()

    assert done["status"] == "completed"
    assert done["answer"] == "The answer was rejected by the human reviewer."


def test_approve_unknown_thread_is_404_and_already_finished_run_is_409():
    client = subgraph_client()
    assert client.post("/approve", json={"thread_id": "nope", "approved": True}).status_code == 404

    paused = client.post("/chat", json={"message": "x"}).json()
    payload = {"thread_id": paused["thread_id"], "approved": True}
    assert client.post("/approve", json=payload).status_code == 200
    assert client.post("/approve", json=payload).status_code == 409  # nothing left to approve


# ───────────────────────────── trace: how the state evolves ─────────────────────────────


def test_chat_trace_shows_each_node_with_what_it_wrote_and_the_state_after():
    body = chat_client("Hello!").post("/chat", json={"message": "hi"}).json()

    steps = [(s["graph"], s["node"]) for s in body["trace"]]
    assert steps == [("main", "(start)"), ("main", "mastermind_agent"), ("main", "chat_agent")]

    start, mastermind, chat = body["trace"]
    assert start["update"] is None and start["state"]["human_message"] == "hi"
    assert mastermind["update"] == {"route": "chat_agent"}             # what the node wrote...
    assert mastermind["state"]["route"] == "chat_agent"                # ...and the state after it
    assert chat["update"] == {"ai_message": "Hello!"}
    assert chat["state"]["ai_message"] == "Hello!" and chat["state"]["human_message"] == "hi"
    assert [s["index"] for s in body["trace"]] == [1, 2, 3]


def test_subgraph_steps_appear_with_their_own_state():
    body = subgraph_client().post("/chat", json={"message": "go"}).json()
    inner = [s for s in body["trace"] if s["graph"] == "subgraph_agent"]

    assert [s["node"] for s in inner] == ["(start)", "planner_agent", "worker", "worker", "worker", "assemble"]
    assert set(inner[0]["state"]) == {"topic", "answers"}               # SubgraphState keys, not MainState
    assert inner[1]["update"]["plan"]["subtasks"] == ["A", "B", "C"]
    assert len(inner[-1]["state"]["answers"]) == 3 and inner[-1]["state"]["result"]
    main_state_keys = set(body["trace"][0]["state"]) | set(body["trace"][-1]["state"] or {})
    assert not ({"plan", "answers", "topic"} & main_state_keys)         # nothing leaks into the main state
    assert body["trace"][-1]["node"] == "__interrupt__"                 # the pause is part of the trace
    assert "waiting_for_human" in body["trace"][-1]["update"]


def test_approve_trace_covers_only_the_resumed_part():
    client = subgraph_client()
    paused = client.post("/chat", json={"message": "go"}).json()

    done = client.post("/approve", json={"thread_id": paused["thread_id"], "approved": True}).json()

    assert [s["node"] for s in done["trace"]] == ["(start)", "human_approval"]
    assert done["trace"][1]["update"] == {"approval": "approved"}


def test_a_failing_graph_returns_502_with_the_reason():
    chat = ChatAgent()

    def boom(messages):
        raise RuntimeError("LLM provider is down")

    chat.llm = type("BrokenLLM", (), {"invoke": staticmethod(boom)})()
    graph = build_main_graph({"mastermind_agent": mastermind_routing_to("chat_agent"), "chat_agent": chat})

    response = TestClient(create_app(graph)).post("/chat", json={"message": "hi"})

    assert response.status_code == 502
    assert "RuntimeError: LLM provider is down" in response.json()["detail"]
