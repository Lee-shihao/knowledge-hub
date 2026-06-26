# Wheel Package + Docker Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build knowledge-hub as a standard Python wheel, install system-wide in Docker, and lower Python requirement to 3.10.

**Architecture:** `uv build --wheel` in builder stage → `uv pip install --system dist/*.whl` in runtime stage → `kh` entrypoint works directly. No venv, no symlinks, no uv at runtime.

**Tech Stack:** hatchling, uv, nvidia/cuda:12.4.0-runtime-ubuntu22.04 (Python 3.10)

## Global Constraints

- Python `>=3.10` in pyproject.toml
- `uv build --wheel` produces `dist/knowledge_hub-0.1.0-py3-none-any.whl`
- `uv pip install --system` in runtime — pip installs to system Python, entrypoint at `/usr/local/bin/kh`
- `kh serve --host 0.0.0.0` as ENTRYPOINT — no uv run, no venv
- `dist/` added to `.dockerignore`
- All existing tests must pass
- `uv.lock` must be regenerated for 3.10

---

### Task 1: pyproject.toml — Python >=3.10 + regenerate uv.lock

**Files:**
- Modify: `pyproject.toml:11`
- Modify: `uv.lock` (regenerated)

**Interfaces:**
- Consumes: nothing
- Produces: `requires-python = ">=3.10"`, updated `uv.lock`

- [ ] **Step 1: Change requires-python**

Edit `pyproject.toml` line 11:

```
# Before:
requires-python = ">=3.12"
# After:
requires-python = ">=3.10"
```

- [ ] **Step 2: Regenerate uv.lock**

Run: `uv lock`
Expected: uv re-resolves dependency tree for 3.10, updates uv.lock

- [ ] **Step 3: Verify tests still pass**

Run: `.venv/bin/pytest tests/ -v -k "not (integration_setup or qdrant_available)" 2>&1 | tail -5`
Expected: Pass count unchanged from baseline

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: lower Python requirement to >=3.10, regenerate uv.lock

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Dockerfile — rewrite with wheel build + system install

**Files:**
- Modify: `Dockerfile`

**Interfaces:**
- Consumes: `pyproject.toml`, `uv.lock`, `src/`, `CLAUDE.md` (from Task 1)
- Produces: Docker image with `kh` command at `/usr/local/bin/kh`

- [ ] **Step 1: Replace Dockerfile**

Replace the entire content of `Dockerfile`:

```dockerfile
# Build:
#   docker build -t knowledge-hub .
#
# Builder: produce standard wheel via uv build.
# Runtime: system-level pip install → kh entrypoint works directly.
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY CLAUDE.md ./
COPY src/ ./src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv build --wheel

# ============================================================
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY --from=builder /app/dist/*.whl /tmp/

RUN uv pip install --system /tmp/knowledge_hub-0.1.0-py3-none-any.whl \
    && rm /tmp/*.whl

RUN useradd -m -u 1000 kh
USER kh

EXPOSE 8765 8766
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/mcp', data=b'{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}', timeout=3)" || exit 1

ENTRYPOINT ["kh", "serve", "--host", "0.0.0.0"]
```

- [ ] **Step 2: Build image**

Run: `docker build -t knowledge-hub . 2>&1 | tail -5`
Expected: `naming to docker.io/library/knowledge-hub:latest done`

- [ ] **Step 3: Smoke test — check kh works and image size**

Run:
```bash
docker run --rm knowledge-hub kh --help
docker images knowledge-hub --format "{{.Size}}"
```
Expected: CLI help output, image size reported

- [ ] **Step 4: Start container and verify both ports respond**

Run:
```bash
docker run -d --name kh-test -p 18765:8765 -p 18766:8766 knowledge-hub
sleep 5
curl -s http://127.0.0.1:18765/mcp -X POST -H "Content-Type: application/json" -H "Accept: application/json" -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python3 -c "import sys,json; tools=json.load(sys.stdin)['result']['tools']; print(f'{len(tools)} tools registered')"
curl -s -X POST http://127.0.0.1:18766/upload -F "file=@CLAUDE.md"
docker rm -f kh-test
```
Expected: 3 tools registered, upload returns job_id

- [ ] **Step 5: Commit**

```bash
git add Dockerfile
git commit -m "refactor(docker): wheel build + system pip install, no venv symlinks

Builder runs uv build --wheel, runtime does uv pip install --system.
kh entrypoint works directly — no uv run, no symlink permission issues.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: .dockerignore — exclude dist/

**Files:**
- Modify: `.dockerignore`

**Interfaces:**
- Consumes: nothing (standalone)
- Produces: `dist/` excluded from build context

- [ ] **Step 1: Add dist/ to .dockerignore**

Add after the existing `data/` line:

```dockerignore
# Build artifacts
dist/
```

- [ ] **Step 2: Commit**

```bash
git add .dockerignore
git commit -m "chore: add dist/ to .dockerignore

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Final verification

- [ ] **Step 1: Verify all files consistent**

Run:
```bash
grep ">=3.10" pyproject.toml
ls Dockerfile .dockerignore docker-compose.yml
docker images knowledge-hub --format "{{.Size}}"
```

- [ ] **Step 2: Run full test suite**

Run: `.venv/bin/pytest tests/ -v -k "not (integration_setup or qdrant_available)" 2>&1 | tail -5`

- [ ] **Step 3: Commit and push**

```bash
git add -A && git commit -m "chore: final verification — wheel package + Docker" && git push origin dev
```
