# Build GPU (default):
#   docker build -t knowledge-hub:gpu .
#
# Build CPU (smaller, no CUDA bloat):
#   docker build --build-arg BASE_IMAGE=python:3.12-slim \
#                --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu \
#                -t knowledge-hub:cpu .
ARG BASE_IMAGE=nvidia/cuda:12.4.0-runtime-ubuntu22.04
ARG TORCH_INDEX=""

FROM ${BASE_IMAGE}

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Phase 1: Install dependencies only (cached layer, no project install yet)
COPY pyproject.toml uv.lock ./
COPY CLAUDE.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Phase 2: Copy source and install the project itself
COPY src/ ./src/
RUN uv sync --frozen --no-dev

# Phase 3: CPU optimization — replace GPU torch with CPU-only torch
# Saves ~3GB by dropping nvidia-cublas/cudnn/cufft/nccl/… and triton
ARG TORCH_INDEX
RUN if [ -n "$TORCH_INDEX" ]; then \
        uv pip install --index-url "$TORCH_INDEX" torch --reinstall \
        && uv pip freeze | grep -E "^nvidia-|^triton" | cut -d= -f1 | xargs -r uv pip uninstall -y; \
    fi

# Non-root user
RUN useradd -m -u 1000 kh && chown -R kh:kh /app
USER kh

EXPOSE 8765 8766

# --start-period=120s gives models time to load before health checking begins
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/mcp', data=b'{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}', timeout=3)" || exit 1

ENTRYPOINT ["uv", "run", "kh", "serve", "--host", "0.0.0.0"]
