"""Pick the LLM from settings (LLM_PROVIDER=groq|fake)."""

import logging

from app.config import Settings
from app.llm.base import LLMClient
from app.llm.fake import ExtractiveLLM
from app.llm.groq import GroqLLM

log = logging.getLogger(__name__)


def make_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "fake":
        log.info("LLM provider: fake (extractive, no network)")
        return ExtractiveLLM()
    if not settings.groq_api_key:
        log.warning("LLM_PROVIDER=groq but GROQ_API_KEY is not set: /ask will return try_again")
    return GroqLLM(settings.groq_api_key, settings.groq_model, settings.llm_timeout_s)
