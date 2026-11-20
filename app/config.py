"""Application settings, loaded from environment / .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: Literal["groq", "fake"] = "groq"  # D18: "fake" runs everything without a key
    groq_api_key: str = ""
    groq_model: str = "qwen/qwen3.8-27b"  # D45; llama-3.1-8b-instant was retired

    jwt_secret: str = "dev-only-secret-change-me-32bytes-min!"
    jwt_issuer: str = "moveinsync-dev"
    jwt_algorithm: str = "HS256"
    jwt_ttl_minutes: int = 60

    # D20: one request-level budget every step draws from; repair call only if enough is left.
    request_deadline_s: float = 2.8
    repair_min_remaining_s: float = 1.0
    llm_min_remaining_s: float = 0.3  # below this, don't even start an LLM call
    llm_timeout_s: float = 2.5  # hard cap per call (also bounded by the remaining deadline)
    # D41/D58: generation settings. Reasoning models (e.g. gpt-oss) need more max_tokens.
    llm_max_tokens: int = 300
    llm_temperature: float = 0.0

    # D22: candidate pool from the vector store, top-k of it sent to the LLM (= max citations).
    retrieval_pool: int = 8
    retrieval_top_k: int = 3
    # D56/D57: "hybrid" = vector + BM25 fused with RRF (k = 60). tau always uses the vector score.
    retrieval_mode: Literal["vector", "hybrid"] = "vector"
    rrf_k: int = 60
    # D60: cross-encoder reranks ALL chunks of the active collection (cap 40); tau unchanged.
    rerank: Literal["off", "on"] = "off"
    rerank_max_candidates: int = 40
    reranker_model_dir: Path = Path("models/ms-marco-MiniLM-L6-v2")
    # D61: on covered=false, retrieve once more with the model's search_query (answer call #2
    # answers the ORIGINAL question). Only if round 1 used 1 call and >= 1.2 s is left.
    rewrite_on_miss: Literal["off", "on"] = "off"
    rewrite_min_remaining_s: float = 1.2
    # D5/D52: tau, the off-topic gate on best cosine similarity. From the eval sweep (D50):
    # highest tau with zero false refusals. Margin is thin (0.017); see EVAL_REPORT §4.
    retrieval_min_score: float = 0.38

    data_dir: Path = Path("var")
    policies_dir: Path = Path("data/policies")

    # D33/D58: roles a token may carry; anything else -> 403. Env: ROLES='["employee","admin"]'
    roles: set[str] = {"employee", "admin"}

    # Server-side allow-list of tenants: company_id -> display name in the policy title.
    # Collection names are only ever built from these ids.
    companies: dict[str, str] = {
        "nimbus": "Nimbus Software",
        "kestrel": "Kestrel Bank",
        "saffron": "Saffron Pharma",
    }

    # MiniLM truncates at 256 tokens; 150 words + context header stays safely under it.
    chunk_max_words: int = 150
    ingest_batch_size: int = 8
    max_document_chars: int = 200_000

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def seed_dir(self) -> Path:
        return self.policies_dir / "seed"

    @property
    def updates_dir(self) -> Path:
        return self.policies_dir / "updates"

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "registry.sqlite3"


@lru_cache
def get_settings() -> Settings:
    return Settings()
