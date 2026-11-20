# ---- runtime (default for `docker compose up`): app only, no dev dependencies
FROM python:3.12-slim AS runtime

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

# Bake the ONNX MiniLM embedding model (~80 MB) into the image so the first question is not
# a cold download (would blow the 3 s budget) and the container works offline.
RUN uv run --no-sync python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as E; E()(['warmup'])"

COPY app ./app
COPY scripts ./scripts
COPY eval ./eval
COPY data ./data
RUN uv sync --frozen --no-dev

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---- test: runtime + dev dependencies + tests. Not started by `docker compose up`:
#   docker compose --profile test run --rm tests
FROM runtime AS test
RUN uv sync --frozen
COPY tests ./tests
CMD ["uv", "run", "--no-sync", "pytest"]
