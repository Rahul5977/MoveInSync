"""LLM abstraction so tests/eval can use a fake and the demo uses Groq."""

from typing import Protocol


class LLMError(Exception):
    """Any LLM failure: timeout, 429 quota, 5xx, network, missing key. Caller maps to TRY_AGAIN."""


class TruncatedOutput(Exception):
    """The model stopped because it hit max_tokens (finish_reason == "length").

    Not a provider failure: the answerer treats it as invalid output (repair call or try_again),
    so a cut-off JSON is never half-parsed (D41).
    """

    def __init__(self, partial: str) -> None:
        super().__init__("truncated output (finish_reason=length)")
        self.partial = partial


class LLMClient(Protocol):
    provider: str  # "groq" | "fake"
    ready: bool  # False when a call is certain to fail (e.g. no API key); shown on /health

    def complete_json(self, system: str, user: str, timeout_s: float) -> str:
        """Return raw model text expected to be a JSON object.

        Raises LLMError (provider failure) or TruncatedOutput (hit the token limit).
        """
        ...
