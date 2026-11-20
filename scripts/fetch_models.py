"""Download pinned model files and verify SHA-256 (D60).

Used at Docker build time (the build fails on any mismatch) and once locally:
    uv run python -m scripts.fetch_models [--dir models/ms-marco-MiniLM-L6-v2]
Files already present with the right hash are not downloaded again.
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from app.rag.reranker import CE_FILES, CE_REPO, CE_REVISION


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(target: Path, repo: str = CE_REPO, revision: str = CE_REVISION, files=None) -> None:
    files = files or CE_FILES
    target.mkdir(parents=True, exist_ok=True)
    for local, (remote, expected) in files.items():
        path = target / local
        if not (path.is_file() and sha256(path) == expected):
            url = f"https://huggingface.co/{repo}/resolve/{revision}/{remote}"
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, path)
        got = sha256(path)
        if got != expected:
            path.unlink()
            sys.exit(f"SHA-256 mismatch for {local}: got {got}, expected {expected}")
        print(f"ok {local} {got}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("models/ms-marco-MiniLM-L6-v2"))
    fetch(ap.parse_args().dir)
