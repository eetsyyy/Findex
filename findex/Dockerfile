# Run from the nested findex/ project directory (docker build -t findex-web .).
FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src/ src/
COPY README.md ./
COPY scripts/prepare_web_corpus.py scripts/prepare_web_corpus.py
COPY results/lab06/crawl_5.jsonl results/lab06/crawl_5.jsonl
RUN uv sync --frozen --no-dev \
    && .venv/bin/python scripts/prepare_web_corpus.py results/lab06/crawl_5.jsonl --out /app/web_corpus \
    && .venv/bin/findex index /app/web_corpus --out /app/index.bin

FROM python:3.13-slim
ENV PATH="/app/.venv/bin:$PATH" INDEX_PATH=/app/index.bin CORPUS_PATH=/app/web_corpus PORT=8000 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
RUN groupadd -r appuser && useradd -r -g appuser -u 10001 appuser
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/src /app/src
COPY --from=builder --chown=appuser:appuser /app/index.bin /app/index.bin
COPY --from=builder --chown=appuser:appuser /app/web_corpus /app/web_corpus
USER appuser
EXPOSE 8000
CMD ["uvicorn", "findex.web:app", "--host", "0.0.0.0", "--port", "8000"]

