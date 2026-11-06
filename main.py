"""Command-line demo of the template.

    python main.py                       # runs one demo request per route
    python main.py "What is 2 + 3?"      # runs your own request
    python main.py --auto-approve "Answer in parallel: what is a cat, a dog and a bird?"

While the graph runs you see every node that executes (``->``). Nodes of the subgraph are
indented. When the subgraph agent answers, the graph pauses (human-in-the-loop) and asks you
to approve the answer; ``--auto-approve`` answers "yes" for you.
"""
import asyncio
import sys
import uuid

from langgraph.types import Command

from src.graphs.main_graph import build_main_graph

DEMO_REQUESTS = [
    "Hi! What can you do?",                                    # -> chat_agent
    "What is 1250 + 4750? Also, what time is it?",             # -> tool_agent (tool loop)
    "Reverse the text 'langgraph'",                            # -> mcp_agent (MCP server)
    "Answer in parallel: what is a cat, a dog and a bird?",    # -> subgraph_agent (+ human approval)
]


async def run_request(graph, message: str, auto_approve: bool = False) -> None:
    """Run one request, streaming node updates and handling the human-approval pause."""
    print(f"\n{'=' * 70}\nUSER: {message}\n{'=' * 70}")

    # thread_id identifies the conversation in the checkpointer. A new id = a fresh run.
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    payload = {"human_message": message}

    while True:
        pending_question = None

        # stream_mode="updates": one event per executed node, with the update it returned.
        # subgraphs=True also yields the events of subgraphs. `namespace` is empty for the main
        # graph and looks like ("subgraph_agent:<run id>",) for a node running inside a subgraph.
        async for namespace, update in graph.astream(payload, config, stream_mode="updates", subgraphs=True):
            parent = " > ".join(part.split(":")[0] for part in namespace)
            for node, delta in update.items():
                if node == "__interrupt__":
                    pending_question = delta[0].value
                elif parent:
                    print(f"  -> {node}   (inside {parent})")
                else:
                    print(f"-> {node}")

        if pending_question is None:
            break  # the graph reached END

        # The graph is paused at `interrupt`. Resume it with the human's answer.
        print(f"\n[PAUSED] {pending_question['question']}\n{pending_question['preview']}\n")
        answer = "yes" if auto_approve else input("Your answer: ")
        payload = Command(resume=answer)

    final_state = (await graph.aget_state(config)).values
    print(f"\nROUTE : {final_state.get('route')}")
    print(f"\nASSISTANT:\n{final_state.get('ai_message')}")


async def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    auto_approve = "--auto-approve" in sys.argv

    graph = build_main_graph()
    for message in args or DEMO_REQUESTS:
        # Without a message on the command line the demo must not stop to ask a human.
        await run_request(graph, message, auto_approve=auto_approve or not args)


if __name__ == "__main__":
    asyncio.run(main())
