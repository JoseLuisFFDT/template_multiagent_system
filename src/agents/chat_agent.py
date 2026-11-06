from typing import List

from src.core.agent_registry import AgentRegistry
from src.core.base_agent import BaseAgent
from src.pydantic_models.graph_states.main_state import MainState


@AgentRegistry.register
class ChatAgent(BaseAgent):
    """The simplest possible agent: one LLM call, free-text answer.

    It is also the mastermind's fallback when no specialised agent matches. Use it as the
    smallest example of how an agent reads the state and returns an update.
    """

    name: str = "chat_agent"
    domain: List[str] = ["mastermind"]
    description: str = "Handles greetings, small talk and anything that no specialised agent covers."

    def ask(self, state: MainState) -> dict:
        response = self.llm.invoke(self.build_messages(state.human_message))
        return {"ai_message": response.content}
