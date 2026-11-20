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
from app.rag.bm25 import Bm25Cache
from app.rag.prompt import build_repair_prompt, build_user_prompt, system_prompt
from app.rag.reranker import Reranker
from app.rag.retriever import search
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
    rerank_scores: dict[str, float] | None = None  # D60: cross-encoder logit per chunk id
    # D61, only when REWRITE_ON_MISS=on and round 1 said covered=false:
    search_query: str | None = None  # the model's search phrase (never shown, never prompted)
    rewrite_triggered: bool = False  # True only if answer call #2 was made
    rewrite_skip_reason: str | None = None  # why round 2 did not run, if it didn't
    round2_retrieved_ids: list[str] = field(default_factory=list)
    round2_sent_ids: list[str] = field(default_factory=list)


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


# ---------------------------------------------------------------- one question, step by step
#
# answer_question() reads top to bottom as the pipeline. Each step is a small function; any step
# can end the request early by raising Stop(status, gate). To change one behaviour in the viva,
# edit one step:
#   _retrieve        which chunks we search (company from the JWT, active version only)
#   _tau_gate        "clearly off-topic" -> not covered with 0 LLM calls
#   _call_llm        the LLM call, plus at most one repair call or one connection retry
#   _valid_citations which cited ids we accept (only ids we sent, at most top_k)


class Stop(Exception):
    """Ends the request early with a final status (not_covered / try_again)."""

    def __init__(self, status: AnswerStatus, gate: str, error: str | None = None) -> None:
        super().__init__(gate)
        self.status, self.gate, self.error = status, gate, error


def not_covered(gate: str) -> Stop:
    return Stop(AnswerStatus.NOT_COVERED, gate)


def try_again(gate: str, error: str) -> Stop:
    return Stop(AnswerStatus.TRY_AGAIN, gate, error)


@dataclass
class Ask:
    """State of one question as it moves through the steps."""

    settings: Settings
    deadline: Deadline
    trace: Trace = field(default_factory=Trace)
    llm_calls: int = 0
    version: int | None = None

    def response(self, status: AnswerStatus, gate: str, answer: str, citations=()):
        self.trace.gate = gate
        meta = AnswerMeta(
            llm_calls=self.llm_calls,
            latency_ms=self.deadline.elapsed_ms(),
            policy_version=self.version,
            retrieval_mode=self.settings.retrieval_mode,
        )
        resp = AskResponse(status=status, answer=answer, citations=list(citations), meta=meta)
        return resp, self.trace


def answer_question(
    principal: Principal,
    question: str,
    registry: Registry,
    store: VectorStore,
    llm: LLMClient,
    settings: Settings,
    clock: Clock = time.monotonic,
    bm25: Bm25Cache | None = None,
    reranker: Reranker | None = None,
) -> tuple[AskResponse, Trace]:
    ask = Ask(settings, Deadline(settings.request_deadline_s, clock))
    try:
        pool = _retrieve(ask, principal.company_id, question, registry, store, bm25, reranker)
        sent = pool[: settings.retrieval_top_k]  # 2. only the top k go to the model
        ask.trace.sent_ids = list(dict.fromkeys(c.chunk_id for c in sent))
        out = _call_llm(ask, build_user_prompt(question, sent), llm)  # 3
        if not out.covered:
            if settings.rewrite_on_miss != "on":
                raise not_covered("llm_not_covered")
            # D61: one more search with the model's phrase, answering the ORIGINAL question
            out, sent = _rewrite_round(
                ask, out, principal.company_id, question, sent, llm, registry, store, bm25, reranker
            )
        citations = _valid_citations(ask, out, sent)  # 4
    except Stop as stop:
        if stop.status is AnswerStatus.TRY_AGAIN:
            ask.trace.error = stop.error
            log.warning("try_again (%s): %s", stop.gate, stop.error)
            return ask.response(stop.status, stop.gate, TRY_AGAIN_MSG)
        return ask.response(stop.status, stop.gate, NOT_COVERED_MSG)

    # We escape & < > in the prompt, so the model may echo &amp; etc. (D36).
    answer = html.unescape(out.answer.strip())
    return ask.response(AnswerStatus.ANSWERED, "answered", answer, citations)


def _retrieve(
    ask: Ask, company_id, question, registry, store, bm25, reranker=None
) -> list[RetrievedChunk]:
    """Step 1: search ONLY this company's active collection, then apply the tau gate."""
    s = ask.settings
    reranker = reranker if s.rerank == "on" else None  # D60
    try:
        pool, ask.version, best_vector = search(
            company_id,
            question,
            registry,
            store,
            s.retrieval_pool,
            mode=s.retrieval_mode,
            bm25=bm25,
            rrf_k=s.rrf_k,
            reranker=reranker,
        )
    except Exception as exc:  # store unavailable etc. -> user gets a clear "try again"
        log.exception("Retrieval failed")
        raise try_again("retrieval_error", type(exc).__name__) from exc
    if ask.version is None:
        raise not_covered("no_policy")
    ask.trace.retrieved_ids = [c.chunk_id for c in pool]
    if reranker is not None:
        ask.trace.rerank_scores = {c.chunk_id: round(c.score, 4) for c in pool}
    _tau_gate(ask, pool, best_vector)
    return pool


def _rewrite_round(
    ask: Ask, out: LLMAnswer, company_id, question, sent, llm, registry, store, bm25, reranker
) -> tuple[LLMAnswer, list[RetrievedChunk]]:
    """D61: round 2 after round 1 said covered=false. Exactly one more call, or none.

    Runs only if: a valid search_query came back; round 1 used exactly 1 call (no repair/retry);
    >= rewrite_min_remaining_s is left; and the new top 3 differs from round 1's. The search uses
    the caller's company (JWT) and active collection, like every search; the prompt carries the
    ORIGINAL question. Every way round 2 can fail ends as not_covered (round 1 already said so).
    """
    s, trace = ask.settings, ask.trace
    trace.search_query = out.search_query

    def skip(reason: str) -> Stop:
        trace.rewrite_skip_reason = reason
        return not_covered("llm_not_covered")

    if not out.search_query:
        raise skip("no_search_query")
    if ask.llm_calls != 1:
        raise skip("round1_used_2_calls")
    if ask.deadline.remaining() < s.rewrite_min_remaining_s:
        raise skip("deadline")
    try:
        pool, version, _ = search(
            company_id,
            out.search_query,
            registry,
            store,
            s.retrieval_pool,
            mode=s.retrieval_mode,
            bm25=bm25,
            rrf_k=s.rrf_k,
            reranker=reranker if s.rerank == "on" else None,
        )
    except Exception:
        log.exception("Round-2 retrieval failed")
        raise skip("round2_retrieval_error") from None
    trace.round2_retrieved_ids = [c.chunk_id for c in pool]
    new_sent = pool[: s.retrieval_top_k]
    trace.round2_sent_ids = [c.chunk_id for c in new_sent]
    if version != ask.version or not new_sent:
        raise skip("round2_no_chunks")
    if {c.chunk_id for c in new_sent} == {c.chunk_id for c in sent}:
        raise skip("same_top3")
    if ask.deadline.remaining() < s.rewrite_min_remaining_s:
        raise skip("deadline")

    trace.rewrite_triggered = True
    ask.llm_calls += 1  # call #2; no repair or retry after it (that would be a third call)
    try:
        raw = llm.complete_json(
            _system(s), build_user_prompt(question, new_sent), ask.deadline.remaining()
        )
        out2 = parse_llm_output(raw)
    except Exception as exc:  # timeout, connection, bad JSON, ... -> not covered (fail-safe)
        trace.error = f"round2: {exc}"
        raise not_covered("rewrite_failed") from exc
    if not out2.covered:
        raise not_covered("rewrite_not_covered")
    ask.trace.sent_ids = [c.chunk_id for c in new_sent]
    ask.trace.model_cited_ids = list(out2.cited_ids)
    return out2, new_sent


def _system(s: Settings) -> str:
    return system_prompt(rewrite_on_miss=s.rewrite_on_miss == "on")


def _tau_gate(ask: Ask, pool: list[RetrievedChunk], best_vector: float | None) -> None:
    """Tau gate (D5/D52) on the best VECTOR score (both modes, D56): off-topic -> 0 LLM calls."""
    ask.trace.best_score = best_vector
    if not pool or best_vector is None or best_vector < ask.settings.retrieval_min_score:
        raise not_covered("tau")


def _call_llm(ask: Ask, user_prompt: str, llm: LLMClient) -> LLMAnswer:
    """Step 3: at most 2 calls in total (D6, D20, D53).

    Call 2 happens only as a REPAIR (invalid output) or a RETRY (dropped connection), never both,
    and only if >= repair_min_remaining_s of the deadline is left.
    """
    s, deadline = ask.settings, ask.deadline
    if deadline.remaining() < s.llm_min_remaining_s:
        raise try_again("deadline", "no time left for the LLM call")
    prompt = user_prompt
    while True:
        ask.llm_calls += 1
        try:
            raw = llm.complete_json(_system(s), prompt, deadline.remaining())
            out = parse_llm_output(raw)
            break
        except ConnectionDropped as exc:
            # D53: no answer was generated; retry once (counted) if the slot and time allow.
            if ask.llm_calls >= 2 or deadline.remaining() < s.repair_min_remaining_s:
                raise try_again("llm_error", str(exc)) from exc
            ask.trace.retried = str(exc)
            log.info("retrying LLM call after %s", exc)
        except LLMError as exc:
            raise try_again("llm_error", str(exc)) from exc
        except (InvalidOutput, ModelOutputError) as exc:
            if isinstance(exc, ModelOutputError):
                raw = exc.partial
            if ask.llm_calls >= 2:
                raise try_again("invalid_output", f"invalid after repair: {exc}") from exc
            if deadline.remaining() < s.repair_min_remaining_s:
                raise try_again("invalid_output", f"invalid, no time to repair: {exc}") from exc
            prompt = build_repair_prompt(user_prompt, raw, str(exc))
    ask.trace.model_cited_ids = list(out.cited_ids)
    return out


def _valid_citations(ask: Ask, out: LLMAnswer, sent: list[RetrievedChunk]) -> list[Citation]:
    """Step 4: only ids we sent, deduplicated, at most top_k; text rendered from metadata."""
    by_id = {c.chunk_id: c for c in sent}
    valid = [cid for cid in dict.fromkeys(out.cited_ids) if cid in by_id]
    valid = valid[: ask.settings.retrieval_top_k]
    if not valid:
        raise not_covered("no_valid_citation")
    return [render_citation(by_id[cid]) for cid in valid]
