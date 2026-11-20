"""Embedding models (D59). One small interface, two implementations:

  minilm     Chroma's built-in ONNX all-MiniLM-L6-v2 (the default; same vectors as before D59)
  bge-small  BAAI/bge-small-en-v1.5, official ONNX export at a pinned revision, run with
             onnxruntime + tokenizers (no PyTorch). CLS pooling, L2-normalised, 512-token cap.
             Questions get BGE's query instruction; documents never do.

The vector store calls embed_documents() when writing and embed_query() when searching, so the
instruction can only ever reach questions.
"""

from pathlib import Path
from typing import Protocol

import numpy as np

DEFAULT_MODEL = "minilm"  # also what a collection without metadata was embedded with (D59)

BGE_REPO = "BAAI/bge-small-en-v1.5"
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
# local file name -> (path in the HF repo, SHA-256). Checked at Docker build (scripts/fetch_bge.py).
BGE_FILES = {
    "model.onnx": (
        "onnx/model.onnx",
        "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35",
    ),
    "tokenizer.json": (
        "tokenizer.json",
        "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    ),
}
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
BGE_MAX_TOKENS = 512


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class MiniLmEmbedder:
    """Chroma's default embedding function, called explicitly. Queries and documents alike."""

    name = "minilm"

    def __init__(self) -> None:
        self._fn = None

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._fn is None:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            self._fn = DefaultEmbeddingFunction()
        return [np.asarray(v, dtype=np.float32).tolist() for v in self._fn(texts)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


class BgeSmallEmbedder:
    """bge-small-en-v1.5: tokenize, run the ONNX model, take the [CLS] vector, L2-normalise.

    Files load on first use, so constructing it (e.g. to report a model mismatch) needs no files.
    """

    name = "bge-small"

    def __init__(self, model_dir: Path) -> None:
        self._dir = Path(model_dir)
        self._session = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._session is not None:
            return
        import onnxruntime
        from tokenizers import Tokenizer

        missing = [f for f in BGE_FILES if not (self._dir / f).is_file()]
        if missing:
            raise FileNotFoundError(
                f"bge-small files missing in {self._dir}: {missing}. "
                "Fetch them with: uv run python -m scripts.fetch_bge"
            )
        tok = Tokenizer.from_file(str(self._dir / "tokenizer.json"))
        tok.enable_truncation(max_length=BGE_MAX_TOKENS)
        tok.enable_padding(pad_id=0, pad_token="[PAD]")
        self._tokenizer = tok
        self._session = onnxruntime.InferenceSession(
            str(self._dir / "model.onnx"), providers=["CPUExecutionProvider"]
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        self._load()
        enc = self._tokenizer.encode_batch(texts)
        feed = {
            "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in enc], dtype=np.int64),
        }
        hidden = self._session.run(["last_hidden_state"], feed)[0]
        cls = hidden[:, 0, :]  # CLS pooling: the first token's vector
        cls = cls / np.linalg.norm(cls, axis=1, keepdims=True)
        return cls.astype(np.float32).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([BGE_QUERY_INSTRUCTION + text])[0]


def make_embedder(model: str, bge_dir: Path) -> Embedder:
    if model == "minilm":
        return MiniLmEmbedder()
    if model == "bge-small":
        return BgeSmallEmbedder(bge_dir)
    raise ValueError(f"Unknown embedding model {model!r}")
