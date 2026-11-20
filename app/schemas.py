"""Pydantic models at every boundary: HTTP, LLM output, internal chunks."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Principal(BaseModel):
    """Identity extracted from a verified JWT. The ONLY source of company_id."""

    sub: str
    company_id: str
    role: str = "employee"


class AskRequest(BaseModel):
    # extra="forbid": a client sending company_id (or anything else) gets a 422.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=3, max_length=500)


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    NOT_COVERED = "not_covered"
    TRY_AGAIN = "try_again"


class Citation(BaseModel):
    document: str
    section: str
    chunk_id: str


class AnswerMeta(BaseModel):
    llm_calls: int = 0
    latency_ms: int = 0
    policy_version: int | None = None
    retrieval_mode: str = "vector"


class AskResponse(BaseModel):
    status: AnswerStatus
    answer: str
    citations: list[Citation] = Field(default_factory=list, max_length=3)
    meta: AnswerMeta = Field(default_factory=AnswerMeta)


class RetrievedChunk(BaseModel):
    chunk_id: str
    company_id: str
    company_name: str
    version: int
    doc_title: str
    section_id: str
    section_title: str
    text: str
    score: float


class LLMAnswer(BaseModel):
    """Shape the LLM must return. Anything else is rejected (-> repair call or try_again)."""

    model_config = ConfigDict(extra="ignore", strict=True)

    covered: bool
    answer: str = Field(max_length=2000)
    cited_ids: list[str] = Field(default_factory=list, max_length=10)
