"""LLM abstraction so tests/eval can use a fake and the demo uses Groq."""

from typing import Protocol


class LLMError(Exception):
    """Any LLM failure: timeout, 429 quota, 5xx, network, missing key. Caller maps to TRY_AGAIN."""


class LLMClient(Protocol):
    def complete_json(self, system: str, user: str, timeout_s: float) -> str:
        """Return raw model text expected to be a JSON object. Raises LLMError."""
        ...
