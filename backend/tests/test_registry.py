import typing

import src.agents  # noqa: F401  (registers the agents)
from src.agents.subgraph_agent import SubgraphAgent
from src.core.agent_registry import AgentRegistry
from src.core.base_agent import BaseAgent, get_agent_config
from src.pydantic_models.agents.mastermind_agent import RouteDecision


def test_routable_agents_match_the_router_output_model():
    """The Literal in RouteDecision is typed by hand; this keeps it in sync with the registry."""
    literal_values = set(typing.get_args(RouteDecision.model_fields["agent_name"].annotation))
    assert literal_values == set(AgentRegistry.list_all("mastermind"))


def test_subgraph_agents_are_not_routable():
    routable = AgentRegistry.list_all("mastermind")
    for name in AgentRegistry.list_all("subgraph"):
        assert name not in routable


def test_every_agent_with_an_llm_has_prompt_and_model():
    for domain in ("mastermind", "orchestration", "subgraph"):
        for name in AgentRegistry.list_all(domain):
            if name != SubgraphAgent.name:  # the only agent without an LLM
                assert BaseAgent.get_system_prompt(name)
                assert get_agent_config(name)["model"]
