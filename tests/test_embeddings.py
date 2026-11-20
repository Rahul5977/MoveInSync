"""Embedding models and the model-mismatch guard (D59). No network: bge tests use the files
baked into the Docker image (or fetched once with `python -m scripts.fetch_bge`)."""

import sqlite3

import chromadb
import numpy as np
import pytest
from chromadb.config import Settings as ChromaSettings
from fastapi.testclient import TestClient

from app.embeddings import (
    BGE_FILES,
    BGE_QUERY_INSTRUCTION,
    BgeSmallEmbedder,
    MiniLmEmbedder,
)
from app.ingest.chunker import chunk_policy, parse_policy
from app.llm.fake import FakeLLM
from app.main import create_app
from app.store.registry import Registry
from app.store.vector import EMBEDDING_CHANGED_MSG, EmbeddingModelMismatch, VectorStore
from tests.conftest import REPO, read_policy

BGE_DIR = REPO / "models" / "bge-small-en-v1.5"
needs_bge = pytest.mark.skipif(
    not all((BGE_DIR / f).is_file() for f in BGE_FILES),
    reason="bge-small files not present (run: uv run python -m scripts.fetch_bge)",
)


@pytest.fixture(scope="module")
def bge():
    return BgeSmallEmbedder(BGE_DIR)


@needs_bge
def test_bge_vectors_have_384_dims_and_unit_norm(bge):
    vecs = np.array(bge.embed_documents(["Escort after 8 PM.", "Cabs on Saturday need approval."]))
    assert vecs.shape == (2, 384)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    assert np.isclose(np.linalg.norm(bge.embed_query("escort?")), 1.0, atol=1e-5)


@needs_bge
def test_bge_is_deterministic(bge):
    a = bge.embed_query("Do I get a security escort for a 7:45 PM drop?")
    b = bge.embed_query("Do I get a security escort for a 7:45 PM drop?")
    assert a == b


@needs_bge
def test_bge_paraphrase_scores_higher_than_unrelated(bge):
    q = np.array(bge.embed_query("Will I be charged if I miss my cab?"))
    para, unrelated = map(
        np.array,
        bge.embed_documents(
            [
                "A no-show fee is deducted when an employee does not board the booked cab.",
                "The cafeteria serves lunch between noon and 2 PM.",
            ]
        ),
    )
    assert q @ para > q @ unrelated


def test_query_instruction_only_on_questions(monkeypatch):
    seen: list[str] = []
    emb = BgeSmallEmbedder(BGE_DIR)
    monkeypatch.setattr(emb, "_embed", lambda texts: seen.extend(texts) or [[0.0]] * len(texts))
    emb.embed_documents(["§8 Escort after 8 PM."])
    emb.embed_query("escort?")
    assert seen == ["§8 Escort after 8 PM.", BGE_QUERY_INSTRUCTION + "escort?"]


class RecordingEmbedder:
    """Tiny fake: records which texts went through which path; 3-dim constant vectors."""

    name = "recording"

    def __init__(self):
        self.docs: list[str] = []
        self.queries: list[str] = []

    def embed_documents(self, texts):
        self.docs += texts
        return [[1.0, 0.0, 0.0]] * len(texts)

    def embed_query(self, text):
        self.queries.append(text)
        return [1.0, 0.0, 0.0]


def test_store_embeds_chunks_as_documents_and_questions_as_queries(tmp_path):
    emb = RecordingEmbedder()
    store = VectorStore(tmp_path / "chroma", emb)
    chunks = chunk_policy("nimbus", parse_policy(read_policy("seed/nimbus/v1.md")), 150)
    store.write_chunks("nimbus_v1", chunks, batch_size=8)
    store.query("nimbus_v1", "escort?", k=3)
    assert emb.docs == [c.text for c in chunks]
    assert emb.queries == ["escort?"]
    assert store.model_of("nimbus_v1") == "recording"


def test_model_mismatch_refuses_to_start(settings):
    """Volume seeded with minilm, app restarted with EMBEDDING_MODEL=bge-small -> clear refusal."""
    s = settings.model_copy(update={"embedding_model": "bge-small"})
    with pytest.raises(EmbeddingModelMismatch, match=EMBEDDING_CHANGED_MSG):
        with TestClient(create_app(s, llm=FakeLLM([]))):
            pass


def test_model_mismatch_refuses_queries(settings):
    store = VectorStore(settings.chroma_dir, BgeSmallEmbedder(BGE_DIR))
    with pytest.raises(EmbeddingModelMismatch, match="nimbus_v1 was embedded with 'minilm'"):
        store.query("nimbus_v1", "escort?", k=3)


def test_missing_collection_metadata_counts_as_minilm(tmp_path):
    """Collections written before D59 carry no model name; they were all minilm."""
    client = chromadb.PersistentClient(
        path=str(tmp_path / "chroma"), settings=ChromaSettings(anonymized_telemetry=False)
    )
    client.create_collection("nimbus_v1")  # no metadata, like a pre-D59 volume
    store = VectorStore(tmp_path / "chroma", MiniLmEmbedder())
    assert store.model_of("nimbus_v1") == "minilm"
    store.check_model("nimbus_v1")  # no error


def test_old_registry_gets_model_column_defaulting_to_minilm(tmp_path):
    db = tmp_path / "registry.sqlite3"
    with sqlite3.connect(db) as c:  # pre-D59 schema: no embedding_model column
        c.execute(
            "CREATE TABLE policy_versions (company_id TEXT, version INTEGER,"
            " collection_name TEXT, doc_title TEXT, status TEXT, chunk_count INTEGER,"
            " created_at TEXT, PRIMARY KEY (company_id, version))"
        )
        c.execute(
            "INSERT INTO policy_versions VALUES ('nimbus', 1, 'nimbus_v1', 'T', 'active', 10, 'x')"
        )
    reg = Registry(db)
    reg.init_schema()
    assert reg.embedding_model("nimbus", 1) == "minilm"


def test_ingest_records_embedding_model_in_registry(client, token_for):
    r = client.post(
        "/admin/ingest",
        files={"file": ("v2.md", read_policy("updates/nimbus/v2.md"), "text/markdown")},
        headers=token_for("nimbus", "admin"),
    )
    assert r.status_code == 201
    assert client.app.state.registry.embedding_model("nimbus", 2) == "minilm"
