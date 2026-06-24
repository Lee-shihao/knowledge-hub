# MCP Transport 协议切换设计

## 问题

当前 `kh serve` 使用 SSE 传输协议，存在以下问题：

1. **SSE 是长连接**：客户端必须先建立 SSE 流获取 session_id，再通过 POST 发消息，调试困难
2. **curl 无法直接测试**：需要同时保持 SSE 连接和发送消息，无法用简单的 curl 命令调试
3. **Agent 集成受限**：Claude Code、Hermes、OpenClaw、Codex 等 agent 需要简单的 HTTP 请求-响应模式

## 方案

将 MCP 默认传输从 `sse` 切换为 `streamable-http`（stateless + json_response 模式）。

### MCP 传输协议对比

| 协议 | 连接模式 | curl 调试 | Agent 集成 | 适用场景 |
|------|---------|-----------|-----------|---------|
| **stdio** | 本地管道 | ❌ | ✅ Claude Desktop | 本地 AI 客户端 |
| **sse** | 长连接事件流 | ❌ 需保持连接 | ⚠️ 部分支持 | 实时推送（旧版） |
| **streamable-http** | 请求-响应 | ✅ 直接 POST | ✅ 通用 | **远程 API、Agent 集成** |

### StreamableHTTP 两种模式

| 模式 | 特点 | 适用 |
|------|------|------|
| **stateless=True** | 每个请求独立，无 session 状态 | 无状态 API、curl 测试 |
| **stateless=False** | 有 session 跟踪，支持 SSE 通知 | 需要服务端推送的场景 |

**选择 stateless=True**：知识库查询是无状态的请求-响应，不需要服务端推送通知。

### curl 测试示例（切换后）

```bash
# 直接 POST JSON-RPC，无需先建立 SSE 连接
curl -X POST http://192.168.30.125:8765/mcp \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "query_knowledge_base",
      "arguments": {"query": "BCM2835 SPI接口", "top_k": 3}
    }
  }'
```

## 修改范围

### 1. `config.py` — 更新默认传输

```python
# 之前
MCP_TRANSPORT: Literal["sse", "streamable-http"] = "sse"

# 之后
MCP_TRANSPORT: Literal["sse", "streamable-http"] = "streamable-http"
```

### 2. `mcp_server.py` — 传递 stateless 和 json_response 参数

```python
# 之前
mcp.run(
    host=settings.MCP_HOST,
    port=settings.MCP_PORT,
    transport=transport,
)

# 之后
mcp.run(
    host=settings.MCP_HOST,
    port=settings.MCP_PORT,
    transport=transport,
    stateless_http=True,    # 无状态模式，每个请求独立
    json_response=True,     # 返回 JSON 而非 SSE 流
)
```

### 3. README.md / README_CN.md — 更新使用示例

- 更新 curl 测试示例为直接 POST
- 更新 AI 客户端配置示例
- 保留 SSE 选项说明（向后兼容）

### 4. 测试更新

- 更新 `test_cli.py` 中 serve 相关测试
- 添加 curl 风格的集成测试

## 向后兼容

- `MCP_TRANSPORT=sse` 仍然可用，用户可手动切换回 SSE
- SSE 模式下不需要 `stateless_http` 和 `json_response` 参数

## Agent 集成配置

### Claude Code (`.claude/settings.json`)

```json
{
  "mcpServers": {
    "knowledge-hub": {
      "type": "streamable-http",
      "url": "http://192.168.30.125:8765/mcp",
      "headers": {"Authorization": "Bearer your-token"}
    }
  }
}
```

### Hermes / OpenClaw / Codex

```json
{
  "mcpServers": {
    "knowledge-hub": {
      "url": "http://192.168.30.125:8765/mcp",
      "transport": "streamable-http",
      "headers": {"Authorization": "Bearer your-token"}
    }
  }
}
```
