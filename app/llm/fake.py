"""Scriptable fake LLM for tests. Never touches the network."""

from collections.abc import Callable

from app.llm.base import LLMError

Responder = Callable[[str, str], str]


class FakeLLM:
    def __init__(self, responses: list[str | Exception] | Responder | None = None) -> None:
        self._responses = responses
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if callable(self._responses):
            return self._responses(system, user)
        if not self._responses:
            raise LLMError("FakeLLM has no scripted responses left")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt
