import pytest
from fastapi.testclient import TestClient

from app.auth import issue_token
from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def token_for():
    def _make(company: str, role: str = "employee") -> dict[str, str]:
        return {"Authorization": f"Bearer {issue_token(sub='t', company_id=company, role=role)}"}

    return _make
