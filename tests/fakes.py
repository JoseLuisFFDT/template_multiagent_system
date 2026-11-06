"""Test doubles that replace the LLM, so graph logic can be tested without network or cost."""
from langchain_core.messages import AIMessage


class ScriptedLLM:
    """Chat model that returns pre-scripted messages, in order, on every ``invoke``.

    It implements the two methods the agents use: ``invoke`` and ``bind_tools``.
    """

    def __init__(self, responses: list[AIMessage]):
        self.responses = list(responses)
        self.calls: list[list] = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return self.responses.pop(0)


def tool_call_message(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    """An AIMessage in which the LLM asks the application to run a tool."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])
