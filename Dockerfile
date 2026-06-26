# Build:
#   ./build.sh   (or: uv build --wheel && docker build -t knowledge-hub .)
#
# Minimal runtime image — no uv, no build toolchain.
# Uses python:3.12-slim for a small footprint.
FROM python:3.12-slim

# PyPI index URL. Defaults to official PyPI.
# For faster downloads in China, override with:
#   --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_INDEX_URL=

# Copy pre-built wheel from local dist/ (glob matches any version)
COPY dist/knowledge_hub-*.whl /tmp/

# Install the package and its dependencies, then drop the wheel
RUN pip install --no-cache-dir ${PIP_INDEX_URL:+ -i ${PIP_INDEX_URL}} /tmp/knowledge_hub-*.whl \
    && rm -f /tmp/knowledge_hub-*.whl

# Non-root user
RUN useradd -m -u 1000 kh

# Pre-create volume mount points (owned by kh before USER switch)
RUN mkdir -p /app/data /app/storage /app/models && chown -R kh:kh /app

USER kh

EXPOSE 8765 8766
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python3 -c "import urllib.request, os; token = os.environ.get('KH_SERVER_AUTH_TOKEN', ''); req = urllib.request.Request('http://127.0.0.1:8765/mcp', data=b'{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}', headers={'Content-Type': 'application/json', 'Accept': 'application/json'}); req.add_header('Authorization', f'Bearer {token}') if token else None; urllib.request.urlopen(req, timeout=3)" || exit 1

ENTRYPOINT ["kh", "serve", "--host", "0.0.0.0"]
