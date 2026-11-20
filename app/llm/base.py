"""LLM abstraction so tests use a fake and the demo uses Groq."""

from typing import Protocol


class LLMError(Exception):
    """Any LLM failure: timeout, quota, 5xx, network. Caller maps it to TRY_AGAIN."""


class LLMClient(Protocol):
    def complete_json(self, system: str, user: str) -> str:
        """Return raw model text expected to be a JSON object. Raises LLMError."""
        ...
