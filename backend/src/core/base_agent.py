"""Base class shared by every agent of the system.

An *agent* in this template is a plain Python class that:

* declares its identity (``name``, ``description``, ``domain``) as class attributes,
* is configured from ``agents_config.yaml`` (model, temperature, API key, output model),
* owns a system prompt stored in ``src/system_prompts/<name>.py``,
* exposes an ``ask(state)`` method that LangGraph uses **as the node function**.

Keeping the LangGraph wiring (``src/graphs``) separate from the agent logic
(``src/agents``) means an agent can be unit-tested without building any graph.
"""
import functools
import importlib
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from src.config import get_llm

load_dotenv()

_CONFIG_PATH = Path(__file__).parent.parent.parent / "agents_config.yaml"


@functools.lru_cache(maxsize=1)
def _load_config() -> dict[str, Any]:
    """Parse agents_config.yaml once; every agent shares the cached result."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_agent_config(agent_name: str) -> dict[str, Any]:
    """Return the effective config of an agent: the ``defaults`` block overridden by its own block.

    An agent that has no block under ``agents:`` simply uses the defaults.
    """
    config = _load_config()
    overrides = (config.get("agents") or {}).get(agent_name) or {}
    return {**(config.get("defaults") or {}), **overrides}


class BaseAgent(ABC):
    """Abstract base class for all AI agents in the system.

    Subclasses must define the class attributes ``name``, ``description`` and
    ``domain`` and implement :meth:`ask`.
    """

    # Set to False for agents that never call an LLM themselves (e.g. an agent that only
    # wraps a subgraph). Such agents skip the API-key / model / prompt loading.
    requires_llm: bool = True

    def __init__(self):
        agent_name = self.metadata()["name"]
        self.config = get_agent_config(agent_name)
        self._structured_llm = None  # built lazily by structured_invoke()

        if self.requires_llm:
            llm_token = self.get_llm_token(agent_name)
            self.llm = self.get_chat_instance(llm_token, agent_name)
            self.system_prompt = self.get_system_prompt(agent_name)

    @classmethod
    def metadata(cls) -> dict[str, Any]:
        """Identity of the agent, read from class attributes.

        It is a ``classmethod`` so that ``AgentRegistry.register`` can read it from the
        *class*, before any instance exists.
        """
        return {"name": cls.name, "description": cls.description, "domain": cls.domain}

    def structured_invoke(self, messages: list) -> BaseModel:
        """Call the LLM and get back an instance of the agent's ``output_model``.

        ``output_model`` is the dotted path of a Pydantic class declared in
        agents_config.yaml. We use ``method="json_schema"`` (native structured outputs of the
        provider) instead of the default tool-calling mode: it is more reliable because the
        model returns JSON that is validated against the schema directly.

        Args:
            messages: List of messages built by :meth:`build_messages`.

        Returns:
            An instance of the configured output model.
        """
        if self._structured_llm is None:
            output_model_path = self.config.get("output_model")
            if not output_model_path:
                raise ValueError(f"'output_model' not defined for agent '{self.name}' in agents_config.yaml.")
            module_path, class_name = output_model_path.rsplit(".", 1)
            output_model_cls = getattr(importlib.import_module(module_path), class_name)
            self._structured_llm = self.llm.with_structured_output(output_model_cls, method="json_schema")

        return self._structured_llm.invoke(messages)

    def build_messages(self, human_message: str, system_prompt: str | None = None) -> list:
        """Build the standard ``[SystemMessage, HumanMessage]`` list for an LLM call.

        Args:
            human_message: The text the agent must work on.
            system_prompt: Override the agent's prompt (used when the prompt is dynamic,
                e.g. the Mastermind injects the list of available agents).
        """
        return [
            SystemMessage(content=system_prompt or self.system_prompt),
            HumanMessage(content=human_message),
        ]

    @abstractmethod
    def ask(self, state):
        """Process the graph state and return the state update.

        LangGraph calls this method as a node. Rules of thumb:

        * Return **only the keys you changed** (a ``dict``); LangGraph merges it into the
          state using each key's reducer. Do not rebuild the whole state.
        * Return a ``Command`` instead when the node must also decide where to go next.
        * The method may be ``async def`` (needed e.g. for MCP clients); LangGraph supports
          sync and async nodes in the same graph.

        Args:
            state: The state the node receives (main state, subgraph state or a ``Send`` payload).
        """

    @staticmethod
    def get_chat_instance(llm_token: str, agent_name: str):
        """Return the chat model configured for ``agent_name`` (see ``agents_config.yaml``)."""
        config = get_agent_config(agent_name)
        return get_llm(
            model_input=config.get("model"),
            temperature_input=config.get("temperature", 0),
            max_tokens_input=config.get("max_tokens"),
            api_key=llm_token,
        )

    @staticmethod
    def get_system_prompt(agent_name: str) -> str:
        """Load ``PROMPT`` from ``src/system_prompts/<agent_name>.py``.

        Prompts live in Python modules (not in the agent class) so they can be reviewed and
        versioned independently of the code.

        Raises:
            FileNotFoundError: If the prompt module does not exist.
        """
        try:
            module = importlib.import_module(f"src.system_prompts.{agent_name}")
        except ModuleNotFoundError as exc:
            raise FileNotFoundError(
                f"System prompt not found for agent '{agent_name}': src/system_prompts/{agent_name}.py"
            ) from exc
        return module.PROMPT

    @staticmethod
    def get_llm_token(agent_name: str) -> str:
        """Return the API key for an agent.

        ``agents_config.yaml`` stores only the *name* of the environment variable
        (``api_key_env``); the secret itself lives in ``.env``, never in the repository.
        Different agents can use different variables (e.g. one key per team or provider).

        Raises:
            ValueError: If ``api_key_env`` is not configured.
            EnvironmentError: If the variable is not set.
        """
        env_var = get_agent_config(agent_name).get("api_key_env")
        if not env_var:
            raise ValueError(f"'api_key_env' not defined for agent '{agent_name}' in agents_config.yaml.")

        token = os.getenv(env_var)
        if not token:
            raise EnvironmentError(f"'{env_var}' is not set in .env (required for agent '{agent_name}').")
        return token
