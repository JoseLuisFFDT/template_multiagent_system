"""Shared test setup.

The tests run OFFLINE: agents are built with a dummy API key (creating a chat model does not
call the network) and their LLM calls are replaced by the fakes in ``tests/fakes.py``.
"""
import os

# Must be set before the agents are imported/instantiated. A real key in .env is not overridden.
os.environ.setdefault("GROQ_API_KEY", "dummy-key-for-offline-tests")
