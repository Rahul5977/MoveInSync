"""Download bge-small-en-v1.5 at the pinned revision and verify SHA-256 (D59).

Used at Docker build time (the build fails on any mismatch) and once locally:
    uv run python -m scripts.fetch_bge [--dir models/bge-small-en-v1.5]
Files already present with the right hash are not downloaded again.
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from app.embeddings import BGE_FILES, BGE_REPO, BGE_REVISION


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for local, (remote, expected) in BGE_FILES.items():
        path = target / local
        if not (path.is_file() and sha256(path) == expected):
            url = f"https://huggingface.co/{BGE_REPO}/resolve/{BGE_REVISION}/{remote}"
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, path)
        got = sha256(path)
        if got != expected:
            path.unlink()
            sys.exit(f"SHA-256 mismatch for {local}: got {got}, expected {expected}")
        print(f"ok {local} {got}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("models/bge-small-en-v1.5"))
    fetch(ap.parse_args().dir)
