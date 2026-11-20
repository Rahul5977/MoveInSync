"""LLM abstraction so tests/eval can use a fake and the demo uses Groq."""

from typing import Protocol


class LLMError(Exception):
    """Any LLM failure: timeout, 429 quota, 5xx, network, missing key. Caller maps to TRY_AGAIN."""


class ModelOutputError(Exception):
    """The provider worked, but the model's output is unusable.

    Not a provider failure: the answerer treats it as invalid output (repair call if time
    allows, else try_again). `partial` is whatever the model produced, for the repair prompt.
    """

    reason = "invalid model output"

    def __init__(self, partial: str = "") -> None:
        super().__init__(self.reason)
        self.partial = partial


class TruncatedOutput(ModelOutputError):
    """Hit max_tokens (finish_reason == "length"); a cut-off JSON is never half-parsed (D41)."""

    reason = "truncated output (finish_reason=length)"


class JsonValidationFailed(ModelOutputError):
    """Provider-side JSON mode rejected the generation (Groq 400 json_validate_failed; D46)."""

    reason = "provider JSON validation failed"


class LLMClient(Protocol):
    provider: str  # "groq" | "fake"
    ready: bool  # False when a call is certain to fail (e.g. no API key); shown on /health

    def complete_json(self, system: str, user: str, timeout_s: float) -> str:
        """Return raw model text expected to be a JSON object.

        Raises LLMError (provider failure) or ModelOutputError (unusable model output).
        """
        ...
