#!/usr/bin/env bash
# Build knowledge-hub Docker image.
#
# Usage:  ./build.sh
#
# 1. Builds a standard wheel via uv build.
# 2. Builds a minimal Docker image (no uv, no build toolchain in runtime).
set -euo pipefail

cd "$(dirname "$0")"

echo "==> Building wheel..."
uv build --wheel

echo "==> Building Docker image..."
docker build \
    --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    -t knowledge-hub .

echo "==> Done: knowledge-hub"
