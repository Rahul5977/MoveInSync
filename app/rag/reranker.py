"""Cross-encoder reranking (D60): score (question, chunk) pairs, best first.

Model: cross-encoder/ms-marco-MiniLM-L6-v2, its own non-quantised ONNX export at a pinned
revision, run with onnxruntime + tokenizers (no PyTorch). Input = (question, chunk text exactly as
embedded: context header + body), pair-encoded, truncated at 512 tokens; score = raw logit.

Candidates are ALL chunks of the active collection (cap 40), cached per collection name and
rebuilt on activation, like the BM25 cache (D56). tau is not affected: it stays on the MiniLM
score of the original question.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # type-only: keeps this module importable alone (Docker fetch step)
    from app.schemas import RetrievedChunk
    from app.store.vector import VectorStore

CE_REPO = "cross-encoder/ms-marco-MiniLM-L6-v2"
CE_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
# local name -> (path in the HF repo, SHA-256); checked at Docker build (scripts/fetch_models.py)
CE_FILES = {
    "model.onnx": (
        "onnx/model.onnx",
        "5d3e70fd0c9ff14b9b5169a51e957b7a9c74897afd0a35ce4bd318150c1d4d4a",
    ),
    "tokenizer.json": (
        "tokenizer.json",
        "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    ),
}
CE_MAX_TOKENS = 512


class CrossEncoder:
    """Loads the ONNX model and tokenizer once; score() is deterministic (CPU, no dropout)."""

    def __init__(self, model_dir: Path) -> None:
        import onnxruntime
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        missing = [f for f in CE_FILES if not (model_dir / f).is_file()]
        if missing:
            raise FileNotFoundError(
                f"cross-encoder files missing in {model_dir}: {missing}. "
                "Fetch them with: uv run python -m scripts.fetch_models"
            )
        tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        tok.enable_truncation(max_length=CE_MAX_TOKENS, strategy="longest_first")
        tok.enable_padding(pad_id=0, pad_token="[PAD]")
        self._tokenizer = tok
        self._session = onnxruntime.InferenceSession(
            str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"]
        )

    def score(self, question: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        enc = self._tokenizer.encode_batch([(question, t) for t in texts])
        feed = {
            "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in enc], dtype=np.int64),
        }
        logits = self._session.run(["logits"], feed)[0]
        return [float(x) for x in logits[:, 0]]


class Reranker:
    """Cross-encoder + candidate cache: collection_name -> all its chunks (at most `cap`)."""

    def __init__(self, model: CrossEncoder, cap: int = 40) -> None:
        self.model = model
        self.cap = cap
        self._candidates: dict[str, list[RetrievedChunk]] = {}
        self._lock = threading.Lock()

    def build(self, store: VectorStore, name: str) -> list[RetrievedChunk]:
        # Deterministic cap: lowest chunk ids first (never reached with ~22 chunks per policy).
        chunks = sorted(store.all_chunks(name), key=lambda c: c.chunk_id)[: self.cap]
        with self._lock:
            # A company has one live collection; drop its older candidate lists.
            prefix = name.rsplit("_v", 1)[0] + "_v"
            for old in [n for n in self._candidates if n.startswith(prefix) and n != name]:
                del self._candidates[old]
            self._candidates[name] = chunks
        return chunks

    def candidates(self, store: VectorStore, name: str) -> list[RetrievedChunk]:
        with self._lock:
            chunks = self._candidates.get(name)
        return chunks if chunks is not None else self.build(store, name)

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._candidates)

    def rerank(self, question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Chunks sorted by logit (desc; ties by chunk id); each carries its logit as `score`."""
        scores = self.model.score(question, [c.text for c in chunks])
        order = sorted(zip(chunks, scores, strict=True), key=lambda cs: (-cs[1], cs[0].chunk_id))
        return [c.model_copy(update={"score": s}) for c, s in order]
