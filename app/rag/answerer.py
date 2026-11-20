"""Orchestration: retrieve -> threshold -> LLM (<=2 calls) -> validate -> respond."""

from app.llm.base import LLMClient
from app.schemas import AskResponse, Principal
from app.store.registry import Registry
from app.store.vector import VectorStore

NOT_COVERED_MSG = (
    "Your company's transport policy does not cover this. "
    "Please contact the transport desk for help."
)
TRY_AGAIN_MSG = (
    "We couldn't get an answer right now. Please try again in a moment, "
    "or contact the transport desk."
)


def answer_question(
    principal: Principal,
    question: str,
    registry: Registry,
    store: VectorStore,
    llm: LLMClient,
) -> AskResponse:
    raise NotImplementedError
