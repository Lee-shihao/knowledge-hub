---
name: knowledge-hub-upload

description: >
  Upload/import local files into the knowledge-hub knowledge base.
  Use when the user wants to "上传文件到知识库"、"导入文档"、"上传 PDF"、
  "import documents" or similar. This skill only performs ingestion, not
  knowledge retrieval or search.

version: 0.1.0

tags:
  - knowledge-hub
  - upload
  - ingestion
  - documents

required_environment_variables:
  - KNOWLEDGE_HUB_BASE_URL
  - KNOWLEDGE_HUB_TOKEN

instructions: |
  Always use scripts/upload.py.
  Never use curl or manually call the upload API.
  Wait until indexing completes before reporting success.
  For knowledge retrieval, use the knowledge-hub MCP tools instead.

examples:
  - 上传文件到知识库
  - 导入文档
  - 上传 PDF
  - Upload this file
  - Import documents
---

# Knowledge-Hub Upload

Uploads a local file into the knowledge-hub and waits until indexing completes.

## Command

```bash
python3 scripts/upload.py <file_path> [--tags "tag1,tag2"] [--timeout 60]
```

| Argument | Description |
|----------|-------------|
| `file_path` | Local file path |
| `--tags` | Optional comma-separated tags |
| `--timeout` | Wait timeout (default: 60 seconds) |

## Exit Codes

| Code | Meaning | Assistant Response |
|------|---------|--------------------|
| 0 | Indexed successfully | Tell the user the upload succeeded and report the indexed chunk count. |
| 1 | Upload failed | Relay the stderr message. |
| 2 | Indexing failed | Relay the server-side error. |
| 3 | Timed out | Inform the user indexing may still be running and offer a longer timeout. |
| 4 | Missing configuration | Ask the user to configure `KNOWLEDGE_HUB_BASE_URL` and `KNOWLEDGE_HUB_TOKEN`. |

## Notes

- Always use the bundled upload script.
- Do not replace it with manual HTTP requests.
- If outbound access to `KNOWLEDGE_HUB_BASE_URL` is blocked, ask the user to allow outbound access instead of using workarounds.