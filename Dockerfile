# Build:
#   ./build.sh   (or: uv build --wheel && docker build -t knowledge-hub .)
#
# Minimal runtime image — no uv, no build toolchain.
# Uses python:3.12-slim for a small footprint.
FROM python:3.12-slim

# Copy pre-built wheel from local dist/
COPY dist/knowledge_hub-0.1.0-py3-none-any.whl /tmp/

# Install the package and its dependencies, then drop the wheel
RUN pip install --no-cache-dir /tmp/knowledge_hub-0.1.0-py3-none-any.whl \
    && rm -f /tmp/knowledge_hub-0.1.0-py3-none-any.whl

# Non-root user
RUN useradd -m -u 1000 kh
USER kh

EXPOSE 8765 8766
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python3 -c "import urllib.request, os; token = os.environ.get('KH_SERVER_AUTH_TOKEN', ''); req = urllib.request.Request('http://127.0.0.1:8765/mcp', data=b'{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}', headers={'Content-Type': 'application/json', 'Accept': 'application/json'}); req.add_header('Authorization', f'Bearer {token}') if token else None; urllib.request.urlopen(req, timeout=3)" || exit 1

ENTRYPOINT ["kh", "serve", "--host", "0.0.0.0"]
