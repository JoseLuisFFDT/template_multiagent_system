"""LLM factory.

Every agent gets its chat model through :func:`get_llm`, which is called from
``BaseAgent.get_chat_instance`` with the values found in ``agents_config.yaml``.

The template only ships with Groq. To support another provider (Anthropic, OpenAI,
Azure...) add a ``provider`` field to ``agents_config.yaml`` and branch on it here:
this is the single place where a chat-model class is instantiated.
"""
from langchain_groq import ChatGroq


def get_llm(
    model_input: str,
    temperature_input: float = 0,
    max_tokens_input: int | None = None,
    api_key: str | None = None,
) -> ChatGroq:
    """Return a configured chat model.

    Args:
        model_input: Model identifier, e.g. ``"openai/gpt-oss-120b"``. Mandatory.
        temperature_input: Sampling temperature. 0 means (almost) deterministic output.
        max_tokens_input: Maximum tokens in the completion. ``None`` means provider default.
            Note that reasoning models count their hidden reasoning tokens in this limit.
        api_key: Provider API key (resolved by ``BaseAgent.get_llm_token``).

    Raises:
        ValueError: If no model is configured. There is deliberately NO silent fallback
            model: a missing model should fail loudly, not run with an unexpected one.
    """
    if not model_input:
        raise ValueError("No 'model' configured. Set it in agents_config.yaml (defaults or per agent).")

    kwargs = {"api_key": api_key} if api_key else {}
    return ChatGroq(
        model=model_input,
        temperature=temperature_input,
        max_tokens=max_tokens_input,
        **kwargs,
    )
