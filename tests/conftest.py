from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.auth import issue_token
from app.config import Settings
from app.main import create_app

REPO = Path(__file__).resolve().parent.parent
POLICIES = REPO / "data" / "policies"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Fresh, isolated storage per test; real policy documents."""
    return Settings(data_dir=tmp_path / "var", policies_dir=POLICIES)


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as c:  # `with` runs the startup seeding
        yield c


@pytest.fixture
def token_for():
    def _make(company: str, role: str = "employee") -> dict[str, str]:
        return {"Authorization": f"Bearer {issue_token(sub='t', company_id=company, role=role)}"}

    return _make


def read_policy(rel: str) -> str:
    return (POLICIES / rel).read_text()
