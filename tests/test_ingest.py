"""Versioning, atomicity and authorisation of policy ingestion. No LLM involved."""

from fastapi.testclient import TestClient

from app.main import create_app
from app.rag.retriever import retrieve
from app.store.vector import VectorStore
from tests.conftest import read_policy

ESCORT_Q = "Does a woman on a 7:45 PM drop need a security escort?"


def _retrieve(client: TestClient, company: str, q: str = ESCORT_Q):
    st = client.app.state
    return retrieve(company, q, st.registry, st.store, k=3)


def _upload(content: str | bytes, name: str = "policy.md") -> dict:
    return {"file": (name, content, "text/markdown")}


def _ingest(client, token_for, company: str, path: str):
    return client.post(
        "/admin/ingest", files=_upload(read_policy(path)), headers=token_for(company, "admin")
    )


class FailingStore(VectorStore):
    """Real Chroma store that raises on the Nth batch once armed (after startup seeding)."""

    def __init__(self, *args, fail_on_batch: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.armed = False
        self._fail_on = fail_on_batch
        self._batches = 0

    def _add_batch(self, col, batch):
        if self.armed:
            self._batches += 1
            if self._batches == self._fail_on:
                raise OSError("simulated disk failure mid-ingest")
        super()._add_batch(col, batch)


def test_startup_ingests_seed_only(cold_settings):
    with TestClient(create_app(cold_settings)) as client:
        _assert_seed_only(client)


def _assert_seed_only(client):
    st = client.app.state
    assert st.seed_outcomes == {
        "kestrel": "ingested v1",
        "nimbus": "ingested v1",
        "saffron": "ingested v1",
    }
    chunks, version = _retrieve(client, "nimbus")
    assert version == 1
    assert chunks and all(c.chunk_id.startswith("nimbus_v1_") for c in chunks)
    assert "8:00 PM" in chunks[0].text  # v1 escort rule
    assert not st.store.exists("nimbus_v2")  # updates/ is never loaded at startup


def test_admin_ingest_switches_version(client, token_for):
    r = _ingest(client, token_for, "nimbus", "updates/nimbus/v2.md")
    assert r.status_code == 201, r.text
    assert r.json() == {
        "company_id": "nimbus",
        "version": 2,
        "collection": "nimbus_v2",
        "chunk_count": 21,
    }
    assert client.app.state.registry.active("nimbus").version == 2

    chunks, version = _retrieve(client, "nimbus")
    assert version == 2
    assert chunks and all(c.chunk_id.startswith("nimbus_v2_") for c in chunks)
    assert "7:30 PM" in chunks[0].text
    # other companies untouched
    assert client.app.state.registry.active("kestrel").version == 1


def test_failed_ingest_leaves_old_version(settings, token_for):
    store = FailingStore(settings.chroma_dir, fail_on_batch=2)
    with TestClient(create_app(settings, store=store)) as client:
        store.armed = True
        r = _ingest(client, token_for, "nimbus", "updates/nimbus/v2.md")
        assert r.status_code == 500
        assert "previous version still active" in r.json()["detail"]

        reg = client.app.state.registry
        assert reg.active("nimbus").version == 1
        assert dict((v, s) for v, s, _ in reg.history("nimbus")) == {1: "active", 2: "failed"}
        assert not store.exists("nimbus_v2")  # half-written collection cleaned up

        store.armed = False
        chunks, version = _retrieve(client, "nimbus")
        assert version == 1
        assert chunks and all(c.chunk_id.startswith("nimbus_v1_") for c in chunks)

        # a retry after the fault succeeds cleanly
        assert _ingest(client, token_for, "nimbus", "updates/nimbus/v2.md").status_code == 201
        assert reg.active("nimbus").version == 2


def test_restart_keeps_v2(cold_settings, token_for):
    settings = cold_settings
    with TestClient(create_app(settings)) as client:
        assert _ingest(client, token_for, "nimbus", "updates/nimbus/v2.md").status_code == 201

    # "restart": brand-new app on the same storage, startup seeding runs again
    with TestClient(create_app(settings)) as client:
        assert client.app.state.seed_outcomes["nimbus"] == "skipped: already active"
        chunks, version = _retrieve(client, "nimbus")
        assert version == 2
        assert all(c.chunk_id.startswith("nimbus_v2_") for c in chunks)


def test_employee_cannot_ingest(client, token_for):
    r = client.post(
        "/admin/ingest",
        files=_upload(read_policy("updates/nimbus/v2.md")),
        headers=token_for("nimbus", "employee"),
    )
    assert r.status_code == 403
    assert client.app.state.registry.active("nimbus").version == 1


def test_cannot_roll_back_to_older_version(client, token_for):
    assert _ingest(client, token_for, "nimbus", "updates/nimbus/v2.md").status_code == 201
    r = _ingest(client, token_for, "nimbus", "seed/nimbus/v1.md")
    assert r.status_code == 409
    assert client.app.state.registry.active("nimbus").version == 2


def test_admin_cannot_ingest_another_companys_document(client, token_for):
    # Kestrel admin uploads Nimbus's document: company comes from the JWT, title must match.
    r = _ingest(client, token_for, "kestrel", "updates/nimbus/v2.md")
    assert r.status_code == 422
    assert client.app.state.registry.active("kestrel").version == 1
    assert not client.app.state.store.exists("kestrel_v2")


def test_admin_cannot_target_company_via_form_field(client, token_for):
    r = client.post(
        "/admin/ingest",
        files=_upload(read_policy("updates/nimbus/v2.md")),
        data={"company_id": "nimbus"},
        headers=token_for("kestrel", "admin"),
    )
    assert r.status_code == 422
    assert "company_id" in r.json()["detail"]
    assert client.app.state.registry.active("nimbus").version == 1


def test_missing_file_rejected(client, token_for):
    r = client.post("/admin/ingest", data={}, headers=token_for("nimbus", "admin"))
    assert r.status_code == 422


def test_non_utf8_file_rejected(client, token_for):
    r = client.post(
        "/admin/ingest", files=_upload(b"\xff\xfe\x00bad"), headers=token_for("nimbus", "admin")
    )
    assert r.status_code == 422
    assert client.app.state.registry.active("nimbus").version == 1


def test_reingest_same_version_conflicts(client, token_for):
    assert _ingest(client, token_for, "nimbus", "updates/nimbus/v2.md").status_code == 201
    assert _ingest(client, token_for, "nimbus", "updates/nimbus/v2.md").status_code == 409
    assert client.app.state.registry.active("nimbus").version == 2


def test_malformed_document_rejected_old_version_kept(client, token_for):
    r = client.post(
        "/admin/ingest", files=_upload("just some text"), headers=token_for("nimbus", "admin")
    )
    assert r.status_code == 422
    assert client.app.state.registry.active("nimbus").version == 1


def test_unknown_company_token_has_no_policy(client):
    chunks, version = _retrieve(client, "globex")
    assert chunks == [] and version is None


def test_cross_company_retrieval_never_leaks(client):
    for company in ("nimbus", "kestrel", "saffron"):
        for q in (ESCORT_Q, "Who approves an airport trip?", "Can a guest use the cab?"):
            chunks, _ = _retrieve(client, company, q)
            assert chunks and all(c.company_id == company for c in chunks)


def test_crash_mid_ingest_reconciled_on_restart(settings, token_for):
    """Simulate a process kill after half the chunks were written: no except block ran."""
    from app.ingest.chunker import chunk_policy, parse_policy
    from app.store.registry import Registry

    reg = Registry(settings.registry_path)
    store = VectorStore(settings.chroma_dir)
    chunks = chunk_policy("nimbus", parse_policy(read_policy("updates/nimbus/v2.md")), 150)
    reg.record_pending("nimbus", 2, "nimbus_v2", "Employee Transport Policy")
    store.write_chunks("nimbus_v2", chunks[:10], batch_size=8)  # ...then the process "dies"
    assert dict((v, s) for v, s, _ in reg.history("nimbus"))[2] == "pending"

    with TestClient(create_app(settings)) as client:  # restart
        st = client.app.state
        assert st.reconcile_outcomes == {"nimbus_v2": "pending -> failed, collection dropped"}
        assert not st.store.exists("nimbus_v2")
        assert dict((v, s) for v, s, _ in st.registry.history("nimbus")) == {
            1: "active",
            2: "failed",
        }
        chunks, version = _retrieve(client, "nimbus")  # v1 still answers
        assert version == 1 and all(c.chunk_id.startswith("nimbus_v1_") for c in chunks)
        # and the crashed version can be ingested cleanly afterwards
        assert _ingest(client, token_for, "nimbus", "updates/nimbus/v2.md").status_code == 201


def test_clean_startup_reconciles_nothing(client):
    assert client.app.state.reconcile_outcomes == {}
