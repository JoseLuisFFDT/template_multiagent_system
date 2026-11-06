"""Didactic tests: how STATE flows through nodes and edges.

Run them with the printouts visible (the output is the "demo"):

    uv run pytest tests/test_state_flow.py -s -v

Everything runs offline: real graphs and agents, only the LLM calls are scripted.

Mental model used in the class
------------------------------
* The **state** is one shared object. Nodes never talk to each other: they READ the state and
  RETURN a partial update.
* LangGraph **merges** each update into the state, key by key, using the key's **reducer**
  (no reducer = overwrite).
* **Edges** decide who runs next. Conditional edges are plain functions of the state.
* Nodes that run in the same **step** (superstep) see the same state and are merged together.
* A **subgraph** has its OWN state. Nothing crosses the border unless an adapter maps it.

Index
-----
Part 1  MainState:   updates, overwrite vs reducer
Part 2  SubgraphState: its own keys, step by step, Send payloads, the adapter
Part 3  Edges:       execution order, conditional edge, fan-out edge, interrupt
"""
import asyncio
import operator
import sys
import uuid
from pathlib import Path
from typing import Annotated

# Lets this file run with the editor's play button (plain `python`), where the repo root is not
# on sys.path. Under `pytest` this is redundant: pyproject.toml already sets `pythonpath = ["."]`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.types import Command, Send
from pydantic import BaseModel

from src.agents.chat_agent import ChatAgent
from src.agents.planner_agent import PlannerAgent
from src.agents.subgraph_agent import SubgraphAgent
from src.agents.tool_agent import ToolAgent
from src.agents.worker_agent import WorkerAgent
from src.graphs.main_graph import build_main_graph
from src.graphs.subgraph import build_subgraph, fan_out
from src.pydantic_models.agents.planner_agent import Plan
from src.pydantic_models.graph_states.main_state import MainState
from src.pydantic_models.graph_states.subgraph_state import SubgraphState, SubtaskInput
from tests.fakes import ScriptedLLM, tool_call_message
from tests.test_main_graph import mastermind_routing_to


# ───────────────────────────── helpers ─────────────────────────────


def banner(title: str) -> None:
    print(f"\n{'-' * 8} {title} {'-' * 8}")  # ASCII only: Windows consoles (cp1252) reject box characters


def new_config() -> dict:
    # thread_id = the "conversation" the checkpointer stores the state under
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def run_updates(graph, payload, config=None, subgraphs: bool = False) -> list[tuple[str, str, dict]]:
    """Run a graph and return ``[(namespace, node, update), ...]`` in execution order.

    ``stream_mode="updates"`` yields, after every node, EXACTLY what that node returned.
    That is the best way to see "what did this node write?".
    """

    async def scenario():
        events = []
        async for chunk in graph.astream(payload, config or new_config(), stream_mode="updates", subgraphs=subgraphs):
            # subgraphs=True yields (namespace, update); False yields just the update.
            namespace, update = chunk if subgraphs else ((), chunk)
            for node, delta in update.items():
                events.append((" > ".join(p.split(":")[0] for p in namespace), node, delta))
        return events

    return asyncio.run(scenario())


def chat_graph(answer: str = "Hello!"):
    chat = ChatAgent()
    chat.llm = ScriptedLLM([AIMessage(content=answer)])
    return build_main_graph({"mastermind_agent": mastermind_routing_to("chat_agent"), "chat_agent": chat})


def tool_graph():
    tool_agent = ToolAgent()
    tool_agent.llm_with_tools = ScriptedLLM(
        [tool_call_message("add", {"a": 40, "b": 2}), AIMessage(content="The result is 42.")]
    )
    return build_main_graph({"mastermind_agent": mastermind_routing_to("tool_agent"), "tool_agent": tool_agent})


def subgraph_parts(subtasks=("A", "B", "C")):
    """Real planner and worker with scripted LLM calls. Returns (planner, worker)."""
    planner, worker = PlannerAgent(), WorkerAgent()
    planner.structured_invoke = lambda messages: Plan(subtasks=list(subtasks))
    worker.llm = ScriptedLLM([AIMessage(content=f"answer {i}") for i in range(1, len(subtasks) + 1)])
    return planner, worker


# ═════════════════ Part 1 — MainState: updates, overwrite vs reducer ═════════════════


def test_1_a_a_node_returns_only_the_keys_it_changes():
    """Each node returns a PARTIAL update; every other key keeps its value."""
    banner("1.A  what each node writes (stream_mode='updates')")
    graph, config = chat_graph("Hello!"), new_config()

    events = run_updates(graph, {"human_message": "hi"}, config)
    for _, node, update in events:
        print(f"  {node:<18} returned {update}")

    assert events == [
        ("", "mastermind_agent", {"route": "chat_agent"}),  # Command(update=...) -> only `route`
        ("", "chat_agent", {"ai_message": "Hello!"}),       # the chat agent -> only `ai_message`
    ]

    # The full state is the merge of the input + every update. Keys nobody wrote keep their defaults.
    state = asyncio.run(graph.aget_state(config)).values
    print(f"  final state        {state}")
    assert state["human_message"] == "hi"  # written by the CALLER, never touched afterwards
    assert state["route"] == "chat_agent"
    assert state["ai_message"] == "Hello!"
    # Keys that NO node has written yet are simply absent from the snapshot (hence .get, not [...]).
    assert "approval" not in state
    assert state.get("messages", []) == []


def test_1_b_reducer_add_messages_appends_instead_of_overwriting():
    """`messages` has the `add_messages` reducer: updates are APPENDED to the existing list."""
    banner("1.B  reducer add_messages through the tool loop")
    graph, config = tool_graph(), new_config()

    events = run_updates(graph, {"human_message": "what is 40 + 2?"}, config)
    for _, node, update in events:
        if "messages" in update:
            print(f"  {node:<18} returned {len(update['messages'])} message(s): {[m.type for m in update['messages']]}")

    # What each visit RETURNED (small) ...
    returned = [[m.type for m in update["messages"]] for _, node, update in events if "messages" in update]
    assert returned == [["system", "human", "ai"], ["tool"], ["ai"]]

    # ... versus what the STATE holds after the reducer merged them (the whole conversation).
    state = asyncio.run(graph.aget_state(config)).values
    print(f"  state.messages     {[m.type for m in state['messages']]}")
    assert [m.type for m in state["messages"]] == ["system", "human", "ai", "tool", "ai"]


def test_1_c_without_a_reducer_a_key_is_overwritten():
    """Same key written twice, no reducer: the last write wins."""
    banner("1.C  no reducer = overwrite")

    class State(BaseModel):
        value: str = ""

    builder = StateGraph(State)
    builder.add_node("first", lambda s: {"value": "written by first"})
    builder.add_node("second", lambda s: {"value": "written by second"})
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)

    result = builder.compile().invoke({})
    print(f"  value = {result['value']!r}")
    assert result["value"] == "written by second"


def test_1_d_parallel_nodes_need_a_reducer():
    """Two nodes in the SAME step write the same key: no reducer -> error; reducer -> merged.

    This is exactly why `SubgraphState.answers` uses `operator.add`: N parallel workers
    all write it in the same step.
    """
    banner("1.D  parallel writes: error vs reducer")

    def parallel_graph(state_cls):
        builder = StateGraph(state_cls)
        builder.add_node("a", lambda s: {"items": ["from a"]})
        builder.add_node("b", lambda s: {"items": ["from b"]})
        builder.add_edge(START, "a")
        builder.add_edge(START, "b")  # two edges out of START -> a and b run in the same step
        builder.add_edge("a", END)
        builder.add_edge("b", END)
        return builder.compile()

    class WithoutReducer(BaseModel):
        items: list[str] = []

    class WithReducer(BaseModel):
        items: Annotated[list[str], operator.add] = []

    with pytest.raises(InvalidUpdateError):
        parallel_graph(WithoutReducer).invoke({})
    print("  without reducer -> InvalidUpdateError (LangGraph cannot decide who wins)")

    merged = parallel_graph(WithReducer).invoke({})["items"]
    print(f"  with operator.add -> {sorted(merged)}")
    assert sorted(merged) == ["from a", "from b"]


# ═════════════════ Part 2 — SubgraphState: its own state ═════════════════


def test_2_a_main_state_and_subgraph_state_share_no_keys():
    """Two independent schemas. The subgraph cannot see (or pollute) the main state."""
    banner("2.A  the two states")
    main_keys, sub_keys = set(MainState.model_fields), set(SubgraphState.model_fields)
    print(f"  MainState     : {sorted(main_keys)}")
    print(f"  SubgraphState : {sorted(sub_keys)}")
    assert main_keys & sub_keys == set()


def test_2_b_the_subgraph_state_evolves_step_by_step():
    """`stream_mode='values'` yields the WHOLE state after every step: a movie of the subgraph."""
    banner("2.B  SubgraphState after every step")
    planner, worker = subgraph_parts()
    subgraph = build_subgraph({"planner_agent": planner, "worker_agent": worker})

    async def snapshots():
        return [s async for s in subgraph.astream(SubgraphState(topic="demo"), stream_mode="values")]

    steps = asyncio.run(snapshots())
    for i, snap in enumerate(steps):
        print(f"  step {i}: plan={'set' if snap.get('plan') else None}  answers={len(snap.get('answers', []))}  result={'set' if snap.get('result') else None}")

    # step 0: the input         step 1: after planner      step 2: after the 3 parallel workers
    # step 3: after assemble
    assert len(steps) == 4
    assert steps[0].get("plan") is None and steps[0].get("answers", []) == []
    assert steps[1]["plan"].subtasks == ["A", "B", "C"] and steps[1].get("answers", []) == []
    assert len(steps[2]["answers"]) == 3 and steps[2].get("result") is None  # ALL workers merged in ONE step
    assert steps[3]["result"] is not None


def test_2_c_workers_receive_a_private_payload_not_the_whole_state():
    """`Send` hands each worker its own small input (`SubtaskInput`), not `SubgraphState`."""
    banner("2.C  what each worker receives")
    planner, worker = subgraph_parts()
    received = []
    original_ask = worker.ask
    worker.ask = lambda task: (received.append(task), original_ask(task))[1]  # spy; must be set BEFORE build

    subgraph = build_subgraph({"planner_agent": planner, "worker_agent": worker})
    asyncio.run(subgraph.ainvoke(SubgraphState(topic="demo")))

    for payload in received:
        print(f"  worker input: {payload!r}")
    assert len(received) == 3
    assert all(isinstance(p, SubtaskInput) or isinstance(p, dict) for p in received)
    assert sorted(SubtaskInput.model_validate(p).subtask for p in received) == ["A", "B", "C"]
    # Only the field `subtask`: the worker does NOT see `topic`, `plan` or `answers`.
    assert set(SubtaskInput.model_fields) == {"subtask"}


def test_2_d_the_adapter_maps_state_in_and_out_of_the_subgraph():
    """`SubgraphAgent.ask` is the only bridge: MainState -> SubgraphState -> MainState."""
    banner("2.D  the adapter between both states")
    planner, worker = subgraph_parts()
    real_subgraph = build_subgraph({"planner_agent": planner, "worker_agent": worker})

    class SpySubgraph:
        """Records what goes in and out of the subgraph, then delegates to the real one."""

        async def ainvoke(self, state):
            self.entered_with = state
            self.returned = await real_subgraph.ainvoke(state)
            return self.returned

    spy = SpySubgraph()
    graph = build_main_graph(
        {"mastermind_agent": mastermind_routing_to("subgraph_agent"), "subgraph_agent": SubgraphAgent(subgraph=spy)}
    )
    config = new_config()
    events = run_updates(graph, {"human_message": "answer in parallel"}, config)

    print(f"  MainState.human_message  = 'answer in parallel'")
    print(f"  -> SubgraphState         = {spy.entered_with}")
    print(f"  <- subgraph returned keys= {sorted(spy.returned)}")
    adapter_update = next(update for _, node, update in events if node == "subgraph_agent")
    print(f"  adapter wrote to Main    = {list(adapter_update)}")

    assert spy.entered_with.topic == "answer in parallel"      # map IN:  human_message -> topic
    assert set(adapter_update) == {"ai_message"}                # map OUT: result -> ai_message, nothing else
    assert adapter_update["ai_message"] == spy.returned["result"]

    # The subgraph's internals (plan, answers...) never leak into the main state.
    main_state = asyncio.run(graph.aget_state(config)).values
    assert not ({"plan", "answers", "topic", "result"} & set(main_state))


def test_2_e_streaming_with_subgraphs_shows_which_state_each_node_writes():
    """With `subgraphs=True` the events of the inner nodes appear, tagged with their parent."""
    banner("2.E  main graph + subgraph events in one stream")
    planner, worker = subgraph_parts()
    graph = build_main_graph(
        {
            "mastermind_agent": mastermind_routing_to("subgraph_agent"),
            "subgraph_agent": SubgraphAgent(subgraph=build_subgraph({"planner_agent": planner, "worker_agent": worker})),
        }
    )

    events = run_updates(graph, {"human_message": "go"}, subgraphs=True)
    for namespace, node, update in events:
        where = f"(inside {namespace})" if namespace else "(main graph)"
        keys = list(update) if node != "__interrupt__" else "(paused: waiting for the human)"
        print(f"  {node:<16} {where:<26} wrote {keys}")

    inner = [(node, list(update)) for ns, node, update in events if ns == "subgraph_agent"]
    assert inner[0] == ("planner_agent", ["plan"])                      # SubgraphState keys...
    assert [n for n, _ in inner].count("worker") == 3
    assert inner[-1] == ("assemble", ["result"])
    outer = [node for ns, node, _ in events if ns == ""]
    assert outer[:2] == ["mastermind_agent", "subgraph_agent"]          # ...and MainState keys outside


# ═════════════════ Part 3 — Edges: how information decides the path ═════════════════


def test_3_a_execution_order_follows_the_edges():
    """The tool loop, as the sequence of visited nodes."""
    banner("3.A  execution order of the tool loop")
    events = run_updates(tool_graph(), {"human_message": "what is 40 + 2?"})
    order = [node for _, node, _ in events]
    print("  " + " -> ".join(["START"] + order + ["END"]))
    assert order == ["mastermind_agent", "tool_agent", "tools", "tool_agent"]


def test_3_b_a_conditional_edge_is_a_pure_function_of_the_state():
    """`tools_condition` only READS the state: last message has tool_calls? -> 'tools', else END."""
    banner("3.B  conditional edge = function(state) -> next node")
    asks_for_tool = MainState(human_message="x", messages=[HumanMessage("x"), tool_call_message("add", {"a": 1, "b": 2})])
    final_answer = MainState(human_message="x", messages=[HumanMessage("x"), AIMessage(content="3")])

    print(f"  last message has tool_calls -> {tools_condition(asks_for_tool)!r}")
    print(f"  last message is an answer   -> {tools_condition(final_answer)!r}")
    assert tools_condition(asks_for_tool) == "tools"
    assert tools_condition(final_answer) == END


def test_3_c_a_command_updates_state_and_chooses_the_next_node_at_once():
    """The mastermind has NO outgoing edges: its return value carries both things."""
    banner("3.C  Command = update + goto")
    mastermind = mastermind_routing_to("tool_agent")

    command = mastermind.ask(MainState(human_message="what is 1 + 1?"))

    print(f"  goto   = {command.goto!r}")
    print(f"  update = {command.update}")
    assert isinstance(command, Command)
    assert command.goto == "tool_agent"            # where to go next (the "edge")
    assert command.update == {"route": "tool_agent"}  # what to write (the "state update")


def test_3_d_the_fan_out_edge_turns_state_into_parallel_runs():
    """`fan_out` reads `state.plan` and returns one `Send` (node + private payload) per subtask."""
    banner("3.D  fan-out edge: state.plan -> N parallel Sends")
    state = SubgraphState(topic="demo", plan=Plan(subtasks=["A", "B", "C"]))

    sends = fan_out(state)

    for send in sends:
        print(f"  Send(node={send.node!r}, arg={send.arg})")
    assert all(isinstance(s, Send) and s.node == "worker" for s in sends)
    assert [s.arg.subtask for s in sends] == ["A", "B", "C"]


def test_3_e_interrupt_the_state_survives_the_pause():
    """At `interrupt` the run stops, but the checkpointer keeps the state; resume continues from it."""
    banner("3.E  interrupt + checkpointer")
    planner, worker = subgraph_parts()
    graph = build_main_graph(
        {
            "mastermind_agent": mastermind_routing_to("subgraph_agent"),
            "subgraph_agent": SubgraphAgent(subgraph=build_subgraph({"planner_agent": planner, "worker_agent": worker})),
        }
    )
    config = new_config()
    run_updates(graph, {"human_message": "go"}, config)

    paused = asyncio.run(graph.aget_state(config))
    print(f"  paused: next={paused.next}  approval={paused.values.get('approval')}")
    print(f"  question shown to the human: {paused.interrupts[0].value['question']!r}")
    assert paused.next == ("human_approval",)
    assert paused.values.get("approval") is None and paused.values["ai_message"]  # work done, decision pending

    asyncio.run(graph.ainvoke(Command(resume="yes"), config))  # same thread_id: continue where it stopped

    done = asyncio.run(graph.aget_state(config))
    print(f"  resumed: next={done.next}  approval={done.values['approval']}")
    assert done.next == () and done.values["approval"] == "approved"


if __name__ == "__main__":
    # Play button / `python tests/test_state_flow.py`: run this file's tests, printing the demo output.
    raise SystemExit(pytest.main([__file__, "-s", "-v", "-p", "no:cacheprovider"]))
