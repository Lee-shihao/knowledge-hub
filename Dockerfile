# Build GPU (default):
#   docker build -t knowledge-hub:gpu .
#
# Build CPU (smaller, no CUDA bloat):
#   docker build --build-arg BASE_IMAGE=python:3.12-slim \
#                --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu \
#                -t knowledge-hub:cpu .
ARG BASE_IMAGE=nvidia/cuda:12.4.0-runtime-ubuntu22.04
ARG TORCH_INDEX=""

# ============================================================
# Stage 1: Builder — installs deps, compiler, uv cache here
# ============================================================
FROM ${BASE_IMAGE} AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies (uv cache lives on a BuildKit mount — never enters the image)
COPY pyproject.toml uv.lock ./
COPY CLAUDE.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Install project + reinstall CPU torch if requested
COPY src/ ./src/
ARG TORCH_INDEX
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev \
    && if [ -n "$TORCH_INDEX" ]; then \
        uv pip install --index-url "$TORCH_INDEX" torch --reinstall \
        && uv pip freeze | grep -E "^nvidia-|^triton" | cut -d= -f1 | xargs -r uv pip uninstall -y; \
    fi

# ============================================================
# Stage 2: Runtime — only .venv + src, no build artifacts
# ============================================================
FROM ${BASE_IMAGE} AS runtime

# Minimal runtime deps (libgomp1 needed by torch CPU, ca-certificates for HTTPS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only what's needed to run
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

WORKDIR /app

# Non-root user
RUN useradd -m -u 1000 kh && chown -R kh:kh /app
USER kh

EXPOSE 8765 8766

# --start-period=120s gives models time to load before health checking begins
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/mcp', data=b'{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}', timeout=3)" || exit 1

ENTRYPOINT ["uv", "run", "kh", "serve", "--host", "0.0.0.0"]
