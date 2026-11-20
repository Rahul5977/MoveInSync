import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.auth import issue_token
from app.config import Settings
from app.ingest.service import seed_policies
from app.llm.fake import FakeLLM
from app.main import create_app
from app.store.registry import Registry
from app.store.vector import VectorStore

REPO = Path(__file__).resolve().parent.parent
POLICIES = REPO / "data" / "policies"


def make_test_settings(**kw) -> Settings:
    """Settings that ignore the developer's .env: tests never see a real key, tau or secret."""
    return Settings(_env_file=None, llm_provider="fake", groq_api_key="", **kw)


@pytest.fixture(scope="session")
def seeded_template(tmp_path_factory) -> Path:
    """Embed the seed policies ONCE per test session; tests get a private copy."""
    var = tmp_path_factory.mktemp("template") / "var"
    s = make_test_settings(data_dir=var, policies_dir=POLICIES)
    reg = Registry(s.registry_path)
    reg.init_schema()
    outcomes = seed_policies(reg, VectorStore(s.chroma_dir), s)
    assert all(o == "ingested v1" for o in outcomes.values()), outcomes
    return var


@pytest.fixture
def cold_settings(tmp_path: Path) -> Settings:
    """Empty storage: the app's own startup does the seeding (for startup/restart tests)."""
    return make_test_settings(data_dir=tmp_path / "var", policies_dir=POLICIES)


@pytest.fixture
def settings(tmp_path: Path, seeded_template: Path) -> Settings:
    """Private copy of pre-seeded storage (v1 of every company already active)."""
    var = tmp_path / "var"
    shutil.copytree(seeded_template, var)
    return make_test_settings(data_dir=var, policies_dir=POLICIES)


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    # No-reply fake: any LLM call raises LLMError, so a test that reaches the LLM by accident
    # gets try_again instead of silently calling a real model.
    with TestClient(create_app(settings, llm=FakeLLM([]))) as c:  # `with` runs startup
        yield c


@pytest.fixture
def token_for():
    def _make(company: str, role: str = "employee") -> dict[str, str]:
        token = issue_token(sub="t", company_id=company, role=role, settings=make_test_settings())
        return {"Authorization": f"Bearer {token}"}

    return _make


def read_policy(rel: str) -> str:
    return (POLICIES / rel).read_text()
