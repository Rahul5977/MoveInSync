import jwt

from app.config import get_settings


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


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
