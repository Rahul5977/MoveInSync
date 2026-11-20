"""Groq client over its OpenAI-compatible HTTP API (JSON mode, per-call timeout).

Every failure becomes LLMError with a short reason; the answerer turns it into TRY_AGAIN.
A missing key is NOT a startup error (D18): the app still starts, /ask returns try_again.
"""

import httpx

from app.llm.base import LLMError

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqLLM:
    def __init__(
        self, api_key: str, model: str, max_timeout_s: float, client: httpx.Client | None = None
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._max_timeout_s = max_timeout_s
        self._client = client or httpx.Client()  # injectable for tests (httpx.MockTransport)

    def complete_json(self, system: str, user: str, timeout_s: float) -> str:
        if not self._api_key:
            raise LLMError("no_api_key")
        body = {
            "model": self._model,
            "temperature": 0,
            "max_tokens": 300,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            resp = self._client.post(
                GROQ_URL,
                json=body,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=min(timeout_s, self._max_timeout_s),
            )
        except httpx.TimeoutException as exc:
            raise LLMError("timeout") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"network: {type(exc).__name__}") from exc

        if resp.status_code == 429:
            raise LLMError("rate_limited")
        if resp.status_code >= 400:
            raise LLMError(f"http_{resp.status_code}")
        try:
            return resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError("malformed_provider_response") from exc
