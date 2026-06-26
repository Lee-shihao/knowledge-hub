#!/bin/bash
# Download BGE-M3 and BGE-reranker-v2-m3 models to ./models/bge-m3
# using standard HuggingFace cache structure.
#
# Usage: bash scripts/download-models.sh
#
# Prerequisites: pip install huggingface_hub
# Total download: ~9GB

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$PROJECT_DIR/models/bge-m3"

export HF_HOME="$MODEL_DIR"

echo "Downloading models to $HF_HOME ..."
echo ""

echo "[1/2] BAAI/bge-m3 (~6.6GB)"
huggingface-cli download BAAI/bge-m3

echo ""
echo "[2/2] BAAI/bge-reranker-v2-m3 (~2.2GB)"
huggingface-cli download BAAI/bge-reranker-v2-m3

echo ""
echo "Done. Models cached at: $HF_HOME/hub/"
du -sh "$HF_HOME/hub/"
