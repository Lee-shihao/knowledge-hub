# Build:
#   docker build -t knowledge-hub .
#
# Single image — app auto-detects GPU at runtime (EMBED_DEVICE=auto).
# Works on both GPU and CPU machines from the same nvidia/cuda base.
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies (uv cache on BuildKit mount — never enters the image)
COPY pyproject.toml uv.lock ./
COPY CLAUDE.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Install project
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ============================================================
# Runtime
# ============================================================
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 kh

COPY --from=builder --chown=kh:kh /app/.venv /app/.venv
COPY --from=builder --chown=kh:kh /app/src /app/src
COPY --from=builder --chown=kh:kh /app/pyproject.toml /app/pyproject.toml
COPY --from=builder --chown=kh:kh /app/CLAUDE.md /app/CLAUDE.md

ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app
USER kh

EXPOSE 8765 8766
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/mcp', data=b'{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}', timeout=3)" || exit 1

ENTRYPOINT ["kh", "serve", "--host", "0.0.0.0"]
