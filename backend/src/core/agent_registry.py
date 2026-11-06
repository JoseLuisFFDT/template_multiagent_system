"""Central catalogue of the agents that exist in the system."""
from typing import List

from src.core.base_agent import BaseAgent


class AgentRegistry:
    """Registry that tracks all available agents.

    Agents register themselves with the ``@AgentRegistry.register`` class decorator, which
    runs when the module is imported (``src/agents/__init__.py`` imports them all).

    Why it exists: the mastermind must know *which agents it can route to* and *what each one
    does*. Instead of hard-coding that list in a prompt, it asks the registry
    (``describe_all("mastermind")``). Adding a new routable agent = write the class and
    decorate it; no router code or prompt has to change.

    ``domain`` is a list of tags that groups agents:

    * ``"mastermind"`` -> agents the mastermind can route a user request to.
    * ``"subgraph"``   -> internal agents of the subgraph (not routable).
    """

    _agents: dict[str, type[BaseAgent]] = {}

    @classmethod
    def describe_all(cls, domain: str) -> str:
        """Return ``- name: description`` lines for every agent of ``domain`` (for prompts)."""
        return "\n".join(
            f"- {name}: {agent_cls.description}"
            for name, agent_cls in cls._agents.items()
            if domain in agent_cls.domain
        )

    @classmethod
    def list_all(cls, domain: str) -> List[str]:
        """Return the names of every agent registered under ``domain``."""
        return [name for name, agent_cls in cls._agents.items() if domain in agent_cls.domain]

    @classmethod
    def register(cls, agent_cls: type[BaseAgent]) -> type[BaseAgent]:
        """Class decorator: register ``agent_cls`` under its ``name``.

        Re-registering an existing name is ignored, which keeps module reloads harmless.
        """
        name = agent_cls.metadata()["name"]
        if name not in cls._agents:
            cls._agents[name] = agent_cls
        return agent_cls
