import jwt

from tests.conftest import test_settings as get_settings


def test_health(client):
    assert client.get("/health").json() == {
        "status": "ok",
        "llm": {"provider": "fake", "ready": True},
    }


def test_ask_requires_token(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 401


def test_ask_rejects_bad_signature(client):
    wrong_key = "a-different-but-long-enough-secret-key-xx"
    forged = jwt.encode({"sub": "x", "company_id": "nimbus"}, wrong_key, algorithm="HS256")
    r = client.post("/ask", json={"question": "hi"}, headers={"Authorization": f"Bearer {forged}"})
    assert r.status_code == 401


def test_ask_rejects_alg_none(client):
    s = get_settings()
    unsigned = jwt.encode(
        {"sub": "x", "company_id": "nimbus", "iss": s.jwt_issuer, "exp": 9999999999},
        key=None,
        algorithm="none",
    )
    headers = {"Authorization": f"Bearer {unsigned}"}
    r = client.post("/ask", json={"question": "hi"}, headers=headers)
    assert r.status_code == 401


def test_client_cannot_override_company(client, token_for):
    r = client.post(
        "/ask", json={"question": "hi", "company_id": "kestrel"}, headers=token_for("nimbus")
    )
    assert r.status_code == 422


def test_ingest_requires_token(client):
    assert client.post("/admin/ingest", files={"file": ("p.md", "x")}).status_code == 401


def _encode(claims: dict) -> dict[str, str]:
    s = get_settings()
    base = {
        "sub": "x",
        "company_id": "nimbus",
        "role": "employee",
        "iss": s.jwt_issuer,
        "exp": 9999999999,
    }
    token = jwt.encode({**base, **claims}, s.jwt_secret, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


def test_expired_token_rejected(client):
    r = client.post("/ask", json={"question": "hi"}, headers=_encode({"exp": 1}))
    assert r.status_code == 401


def test_wrong_issuer_rejected(client):
    r = client.post("/ask", json={"question": "hi"}, headers=_encode({"iss": "someone-else"}))
    assert r.status_code == 401


def test_missing_role_claim_rejected(client):
    s = get_settings()
    claims = {"sub": "x", "company_id": "nimbus", "iss": s.jwt_issuer, "exp": 9999999999}
    token = jwt.encode(claims, s.jwt_secret, algorithm="HS256")
    r = client.post("/ask", json={"question": "hi"}, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401


def test_unknown_company_forbidden(client, token_for):
    assert (
        client.post("/ask", json={"question": "hi"}, headers=token_for("globex")).status_code == 403
    )


def test_unknown_role_forbidden(client):
    r = client.post("/ask", json={"question": "hi"}, headers=_encode({"role": "superuser"}))
    assert r.status_code == 403


def test_unknown_company_admin_cannot_ingest(client, token_for):
    r = client.post(
        "/admin/ingest", files={"file": ("p.md", "x")}, headers=token_for("globex", "admin")
    )
    assert r.status_code == 403
