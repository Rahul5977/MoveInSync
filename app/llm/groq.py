"""Groq client over its OpenAI-compatible HTTP API, with a hard timeout."""

import httpx

from app.llm.base import LLMError

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqLLM:
    def __init__(self, api_key: str, model: str, timeout_s: float) -> None:
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set")
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_s),
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self._model = model

    def complete_json(self, system: str, user: str) -> str:
        body = {
            "model": self._model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            resp = self._client.post(GROQ_URL, json=body)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"Groq call failed: {type(exc).__name__}") from exc
