"""HTTP API in front of the multi-agent system: the entry point of the application.

    POST /chat      {"message": "..."}                       -> run the main graph
    POST /approve   {"thread_id": "...", "approved": true}   -> resume a run paused for human approval
    GET  /health

The main graph is built once, when the server starts, and shared by every request. Its
checkpointer keeps the state of paused runs, so ``/approve`` can resume the run that ``/chat``
started (they are linked by ``thread_id``).
"""
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from src.api.schemas import ApprovalRequest, ChatRequest, ChatResponse
from src.graphs.main_graph import build_main_graph


def get_graph(request: Request) -> CompiledStateGraph:
    return request.app.state.graph


async def graph_status(graph: CompiledStateGraph, thread_id: str) -> ChatResponse:
    """Read the saved state of a run and describe it: finished, or paused waiting for a human."""
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    route = snapshot.values.get("route")

    if snapshot.next:  # the graph stopped at `interrupt`
        pending = snapshot.interrupts[0].value
        return ChatResponse(
            thread_id=thread_id,
            status="needs_approval",
            route=route,
            question=pending["question"],
            preview=pending["preview"],
        )
    return ChatResponse(thread_id=thread_id, status="completed", route=route, answer=snapshot.values.get("ai_message"))


def create_app(graph: CompiledStateGraph | None = None) -> FastAPI:
    """Build the FastAPI app. Tests pass a graph with a stubbed LLM; production passes nothing."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.graph is None:
            app.state.graph = build_main_graph()
        yield

    app = FastAPI(title="Multi-agent system", lifespan=lifespan)
    app.state.graph = graph

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(body: ChatRequest, graph: CompiledStateGraph = Depends(get_graph)) -> ChatResponse:
        # Every request starts a fresh run: a new thread_id = no memory of earlier requests.
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        await graph.ainvoke({"human_message": body.message}, config)
        return await graph_status(graph, thread_id)

    @app.post("/approve", response_model=ChatResponse)
    async def approve(body: ApprovalRequest, graph: CompiledStateGraph = Depends(get_graph)) -> ChatResponse:
        config = {"configurable": {"thread_id": body.thread_id}}
        snapshot = await graph.aget_state(config)
        if not snapshot.values:
            raise HTTPException(status_code=404, detail="Unknown thread_id.")
        if not snapshot.next:
            raise HTTPException(status_code=409, detail="This run is not waiting for approval.")

        # Resume the paused run: `interrupt()` inside human_approval returns this value.
        await graph.ainvoke(Command(resume="yes" if body.approved else "no"), config)
        return await graph_status(graph, body.thread_id)

    return app


app = create_app()
