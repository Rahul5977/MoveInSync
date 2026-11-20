"""Fake LLMs. Never touch the network.

- FakeLLM (scripted): unit tests. A list of replies/exceptions, or a callable(system, user).
- ExtractiveLLM: runs the app/eval with no key (D18). Cites the top section it was given and
  answers with that section's first sentence. It never says covered=false, so in keyless runs
  only the tau gate can produce "not covered"; the LLM gate needs a real model.
"""

import json
import re
from collections.abc import Callable

from app.llm.base import LLMError
from app.rag.prompt import unescape

Responder = Callable[[str, str], str]

SECTION_RE = re.compile(r'<policy_section id="([^"]+)">\n(.*?)\n</policy_section>', re.DOTALL)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class FakeLLM:
    def __init__(self, responses: list[str | Exception] | Responder | None = None) -> None:
        self._responses = responses
        self.calls: list[tuple[str, str]] = []
        self.provider, self.ready = "fake", True

    def complete_json(self, system: str, user: str, timeout_s: float = 0.0) -> str:
        self.calls.append((system, user))
        if callable(self._responses):
            return self._responses(system, user)
        if not self._responses:
            raise LLMError("FakeLLM has no scripted responses left")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def sections_in(user_prompt: str) -> list[tuple[str, str]]:
    """(chunk_id, unescaped text) for every policy section in a prompt."""
    return [(cid, unescape(text)) for cid, text in SECTION_RE.findall(user_prompt)]


class ExtractiveLLM:
    provider, ready = "fake", True

    def complete_json(self, system: str, user: str, timeout_s: float = 0.0) -> str:
        sections = sections_in(user)
        if not sections:
            return json.dumps({"covered": False, "answer": "", "cited_ids": []})
        chunk_id, text = sections[0]
        body = text.split("\n\n", 1)[-1]  # drop the context header line
        first = SENTENCE_RE.split(body.strip(), maxsplit=1)[0]
        return json.dumps({"covered": True, "answer": first, "cited_ids": [chunk_id]})
