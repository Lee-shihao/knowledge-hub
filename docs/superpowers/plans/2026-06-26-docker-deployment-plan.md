# Docker Containerization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Docker image for knowledge-hub with GPU/CPU dual-mode, persistent storage via volumes, docker-compose as primary deployment method.

**Architecture:** Single Dockerfile with `ARG BASE_IMAGE` for GPU/CPU switch. Two-phase `uv sync` build (deps first, then project). Models mounted at runtime via `HF_HOME=/models`. Non-root `kh` user. HEALTHCHECK with 120s start-period for model warmup.

**Tech Stack:** Docker, nvidia/cuda:12.4.0-runtime-ubuntu22.04, python:3.12-slim, uv, docker-compose

## Global Constraints

- Dockerfile must build with both GPU (`nvidia/cuda:12.4.0-runtime-ubuntu22.04`) and CPU (`python:3.12-slim`) base images via build arg
- `uv sync --frozen --no-dev` with two-phase build: `--no-install-project` first, then full after COPY src
- `CLAUDE.md` must be copied for hatchling build (referenced in pyproject.toml `readme`)
- Non-root user `kh` (uid 1000) must own `/app`
- HEALTHCHECK must have `--start-period=120s`, use `python3`
- docker-compose.yml must NOT have `version:` field
- `KH_SERVER_AUTH_TOKEN` must be passed via `.env` file (`${KH_SERVER_AUTH_TOKEN}`)
- Model path `./models/bge-m3` is relative to project root
- `.dockerignore` must exempt `CLAUDE.md` from `*.md` ignore

---

### Task 1: Dockerfile

**Files:**
- Create: `Dockerfile`

**Interfaces:**
- Consumes: `pyproject.toml`, `uv.lock`, `CLAUDE.md`, `src/`
- Produces: `knowledge-hub:gpu` or `knowledge-hub:cpu` Docker image
- Ports: 8765, 8766
- Entrypoint: `uv run kh serve --host 0.0.0.0`

- [ ] **Step 1: Create Dockerfile**

Create `Dockerfile` at the project root with the exact content from the spec:

```dockerfile
# Build: docker build -t knowledge-hub:gpu .
#        docker build --build-arg BASE_IMAGE=python:3.12-slim -t knowledge-hub:cpu .
ARG BASE_IMAGE=nvidia/cuda:12.4.0-runtime-ubuntu22.04
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

# Non-root user
RUN useradd -m -u 1000 kh && chown -R kh:kh /app
USER kh

EXPOSE 8765 8766

# --start-period=120s gives models time to load before health checking begins
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/mcp', data=b'{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}', timeout=3)" || exit 1

ENTRYPOINT ["uv", "run", "kh", "serve", "--host", "0.0.0.0"]
```

- [ ] **Step 2: Verify Dockerfile syntax**

Run: `docker build --dry-run -f Dockerfile . 2>&1 || true` (if Docker BuildKit supports it)
Alternative: just confirm the file exists and has no syntax issues visible to naked eye.

- [ ] **Step 3: Commit**

```bash
git add Dockerfile
git commit -m "feat: add Dockerfile with GPU/CPU dual-mode support

Two-phase uv sync build, non-root kh user, 120s health start-period.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: docker-compose.yml + .env.example

**Files:**
- Create: `docker-compose.yml`
- Create: `.env.example`

**Interfaces:**
- Consumes: Docker image `knowledge-hub:gpu`
- Produces: Running container with MCP :8765 + HTTP upload :8766
- Volumes: `./storage:/app/storage`, `./data:/app/data`, `./models/bge-m3:/models`

- [ ] **Step 1: Create docker-compose.yml**

Create `docker-compose.yml` at the project root:

```yaml
services:
  knowledge-hub:
    image: knowledge-hub:gpu
    ports:
      - "8765:8765"
      - "8766:8766"
    volumes:
      - ./storage:/app/storage
      - ./data:/app/data
      - ./models/bge-m3:/models
    environment:
      - KH_SERVER_HOST=0.0.0.0
      - KH_QDRANT_PATH=/app/storage/qdrant
      - KH_DATA_DIR=/app/data
      - KH_SERVER_AUTH_TOKEN=${KH_SERVER_AUTH_TOKEN}
      - HF_HOME=/models
    env_file:
      - .env
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

- [ ] **Step 2: Create .env.example**

Create `.env.example` as a template (not tracked by git if `.env` is in gitignore, but `.env.example` is committed as documentation):

```bash
# knowledge-hub server auth token (required for LAN access)
KH_SERVER_AUTH_TOKEN=your-secret-token-here
```

Check that `.env` is gitignored:

Run: `grep "^\.env$" .gitignore || echo "NOT FOUND — add to .gitignore"`

- [ ] **Step 3: Verify docker-compose config**

Run: `docker compose config 2>&1 || true`
Expected: Parses without error (may warn about missing `.env` — that's fine).

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml .env.example
git commit -m "feat: add docker-compose.yml for production deployment

No version field (Compose V2), auth via .env, GPU reserved via nvidia driver,
3 persistent volumes (storage, data, models).

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: .dockerignore + scripts/download-models.sh

**Files:**
- Create: `.dockerignore`
- Create: `scripts/download-models.sh`

**Interfaces:**
- Consumes: project file tree
- Produces: Slim build context (excludes venv, tests, local data, dev artifacts)

- [ ] **Step 1: Create .dockerignore**

Create `.dockerignore` at the project root:

```dockerignore
# Python
.venv/
__pycache__/
*.pyc
*.egg-info/

# Git
.git/

# Tests
tests/
.pytest_cache/

# Local data (mounted at runtime, not baked in)
storage/
data/
models/

# Dev artifacts
.superpowers/
.tokenmiser/
graphify-out/
docs/

# IDE
.vscode/
.idea/
.claude/

# Markdown (except CLAUDE.md — required by hatchling build)
*.md
!CLAUDE.md

# Misc
LICENSE
```

- [ ] **Step 2: Create scripts/download-models.sh**

```bash
mkdir -p scripts
```

Create `scripts/download-models.sh`:

```bash
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
```

Make it executable:

Run: `chmod +x scripts/download-models.sh`

- [ ] **Step 3: Verify .dockerignore excludes the right files**

Run: `tar czf - . --exclude=.git -X <(grep -v '^#' .dockerignore | grep -v '^$' | sed 's|/$||' | sed 's|^|./|') 2>/dev/null | wc -c`
Or simpler: just confirm `.dockerignore` exists and looks right.

- [ ] **Step 4: Commit**

```bash
git add .dockerignore scripts/download-models.sh
git commit -m "feat: add .dockerignore and model download helper script

.dockerignore excludes venv, tests, local data, dev artifacts.
Exempts CLAUDE.md for hatchling build. download-models.sh pre-fetches
~9GB of models into ./models/bge-m3 with standard HF cache structure.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Final verification

- [ ] **Step 1: Verify all 4 files exist**

Run: `ls -la Dockerfile docker-compose.yml .dockerignore scripts/download-models.sh .env.example`

- [ ] **Step 2: Verify .dockerignore exempts CLAUDE.md**

Run: `grep "!CLAUDE.md" .dockerignore`
Expected: `!CLAUDE.md`

- [ ] **Step 3: Verify docker-compose has no version field**

Run: `grep "^version:" docker-compose.yml && echo "ERROR: remove version field" || echo "OK: no version field"`

- [ ] **Step 4: Verify Dockerfile has --start-period=120s**

Run: `grep "start-period=120s" Dockerfile`
Expected: match found

- [ ] **Step 5: Commit and push**

```bash
git add -A
git commit -m "chore: final verification — all Docker files in place"
git push origin dev
```
