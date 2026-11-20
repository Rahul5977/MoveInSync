"""Orchestration (ARCHITECTURE §4, /ask): code decides, the LLM only phrases.

  retrieve pool (active collection only) -> tau gate (0 LLM calls) -> top-k sent as data
  -> LLM call 1 -> validate JSON + citations -> [repair call 2, only if >= 1.0 s left] -> respond
  (call 2 is instead a plain retry if call 1 lost its connection, D53; never both)

Guarantees, all enforced here in code:
- at most 2 LLM calls; at most `retrieval_top_k` (3) sections sent, so at most 3 citations
- citations must be ids we actually sent; document/section text is rendered from stored
  metadata, never from model text
- any LLM failure, invalid output or exhausted deadline -> TRY_AGAIN, never a half answer
"""

import html
import logging
import time
from dataclasses import dataclass, field

from pydantic import ValidationError

from app.config import Settings
from app.deadline import Clock, Deadline
from app.llm.base import ConnectionDropped, LLMClient, LLMError, ModelOutputError
from app.rag.prompt import SYSTEM_PROMPT, build_repair_prompt, build_user_prompt
from app.rag.retriever import retrieve
from app.schemas import (
    AnswerMeta,
    AnswerStatus,
    AskResponse,
    Citation,
    LLMAnswer,
    Principal,
    RetrievedChunk,
)
from app.store.registry import Registry
from app.store.vector import VectorStore

log = logging.getLogger(__name__)

NOT_COVERED_MSG = (
    "Your company's transport policy doesn't cover this. "
    "I can only answer from your own company's current policy. "
    "Please contact the transport desk."
)
TRY_AGAIN_MSG = (
    "We couldn't get an answer right now. Please try again in a moment, "
    "or contact the transport desk."
)


@dataclass
class Trace:
    """Internal record of one question, for logs and the evaluation runner. Not sent to users."""

    gate: str = ""  # no_policy | tau | llm_not_covered | no_valid_citation | answered |
    #                 llm_error | invalid_output | deadline | retrieval_error
    best_score: float | None = None
    retrieved_ids: list[str] = field(default_factory=list)
    sent_ids: list[str] = field(default_factory=list)
    model_cited_ids: list[str] = field(default_factory=list)
    error: str | None = None
    retried: str | None = None  # D53: reason for the one connection-level retry, if any


class InvalidOutput(ValueError):
    pass


def parse_llm_output(raw: str) -> LLMAnswer:
    try:
        out = LLMAnswer.model_validate_json(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        raise InvalidOutput(f"{first['type']} at {first['loc']}") from exc
    if out.covered and not out.answer.strip():
        raise InvalidOutput("covered=true with an empty answer")
    return out


def render_citation(c: RetrievedChunk) -> Citation:
    return Citation(
        document=f"{c.company_name} — {c.doc_title} v{c.version}",
        section=f"§{c.section_id} {c.section_title}",
        chunk_id=c.chunk_id,
    )


def answer_question(
    principal: Principal,
    question: str,
    registry: Registry,
    store: VectorStore,
    llm: LLMClient,
    settings: Settings,
    clock: Clock = time.monotonic,
) -> tuple[AskResponse, Trace]:
    deadline = Deadline(settings.request_deadline_s, clock)
    trace = Trace()
    llm_calls = 0
    version: int | None = None

    def respond(
        status: AnswerStatus, gate: str, answer: str, cites=()
    ) -> tuple[AskResponse, Trace]:
        trace.gate = gate
        meta = AnswerMeta(
            llm_calls=llm_calls, latency_ms=deadline.elapsed_ms(), policy_version=version
        )
        return AskResponse(status=status, answer=answer, citations=list(cites), meta=meta), trace

    def not_covered(gate: str):
        return respond(AnswerStatus.NOT_COVERED, gate, NOT_COVERED_MSG)

    def try_again(gate: str, error: str):
        trace.error = error
        log.warning("try_again (%s): %s", gate, error)
        return respond(AnswerStatus.TRY_AGAIN, gate, TRY_AGAIN_MSG)

    # 1. Retrieval: company from the JWT only; active collection only.
    try:
        pool, version = retrieve(
            principal.company_id, question, registry, store, settings.retrieval_pool
        )
    except Exception as exc:  # store unavailable etc. -> user gets a clear "try again"
        log.exception("Retrieval failed")
        return try_again("retrieval_error", type(exc).__name__)
    if version is None:
        return not_covered("no_policy")
    trace.retrieved_ids = [c.chunk_id for c in pool]

    # 2. Tau gate: clearly off-topic -> not covered with 0 LLM calls.
    trace.best_score = pool[0].score if pool else None
    if not pool or pool[0].score < settings.retrieval_min_score:
        return not_covered("tau")

    # 3. Send the top-k as data.
    sent = pool[: settings.retrieval_top_k]
    by_id = {c.chunk_id: c for c in sent}
    trace.sent_ids = list(by_id)
    user_prompt = build_user_prompt(question, sent)

    # 4. LLM call 1 (+ at most one repair call, only if the deadline allows).
    if deadline.remaining() < settings.llm_min_remaining_s:
        return try_again("deadline", "no time left for the LLM call")
    prompt = user_prompt
    while True:
        llm_calls += 1
        try:
            raw = llm.complete_json(SYSTEM_PROMPT, prompt, deadline.remaining())
            out = parse_llm_output(raw)
            break
        except ConnectionDropped as exc:
            # D53: no answer was generated; retry once (counted) if the slot and time allow.
            if llm_calls >= 2 or deadline.remaining() < settings.repair_min_remaining_s:
                return try_again("llm_error", str(exc))
            trace.retried = str(exc)
            log.info("retrying LLM call after %s", exc)
        except LLMError as exc:
            return try_again("llm_error", str(exc))
        except (InvalidOutput, ModelOutputError) as exc:
            if isinstance(exc, ModelOutputError):
                raw = exc.partial
            if llm_calls >= 2:
                return try_again("invalid_output", f"invalid after repair: {exc}")
            if deadline.remaining() < settings.repair_min_remaining_s:
                return try_again("invalid_output", f"invalid, no time to repair: {exc}")
            prompt = build_repair_prompt(user_prompt, raw, str(exc))

    trace.model_cited_ids = list(out.cited_ids)
    if not out.covered:
        return not_covered("llm_not_covered")

    # 5. Citations: only ids we sent, deduplicated, at most top_k; rendered from metadata.
    valid = [cid for cid in dict.fromkeys(out.cited_ids) if cid in by_id][
        : settings.retrieval_top_k
    ]
    if not valid:
        return not_covered("no_valid_citation")
    return respond(
        AnswerStatus.ANSWERED,
        "answered",
        # We escape & < > in the prompt, so the model may echo &amp; etc. (D36).
        html.unescape(out.answer.strip()),
        [render_citation(by_id[cid]) for cid in valid],
    )
