"""Application settings, loaded from environment / .env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    jwt_secret: str = "dev-only-secret-change-me-32bytes-min!"
    jwt_issuer: str = "moveinsync-dev"
    jwt_algorithm: str = "HS256"
    jwt_ttl_minutes: int = 60

    llm_timeout_s: float = 2.0
    retrieval_top_k: int = 3
    retrieval_min_score: float = 0.35

    data_dir: Path = Path("var")
    policies_dir: Path = Path("data/policies")

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
