---
name: knowledge-hub-upload
description: Use when the user wants to upload/import a file into the knowledge-hub knowledge base via its HTTP API. Triggers on phrases like "上传文件到知识库", "导入文档到知识库", "upload to knowledge hub", "import this file into the knowledge base". Do not use for querying — that's handled by knowledge-hub MCP tools.
license: MIT
---

# Knowledge-Hub File Upload

Uploads a file to the Knowledge Hub (local RAG platform) via HTTP API and polls until indexing is complete.

This skill is **only for ingestion**. Querying the knowledge base should use the knowledge-hub MCP tools.

## Configuration (ask user if not set)
| Variable                | Example                        | Description                    |
|-------------------------|--------------------------------|--------------------------------|
| `KNOWLEDGE_HUB_BASE_URL`| `http://192.168.30.125:8766`  | Upload server address          |
| `KNOWLEDGE_HUB_TOKEN`   | `test-token-123`               | Authentication token           |

## Step-by-Step Usage

### 1. Upload File
```bash
curl -s -X POST "$KNOWLEDGE_HUB_BASE_URL/upload" \
  -H "Authorization: Bearer $KNOWLEDGE_HUB_TOKEN" \
  -F "file=@/path/to/your/document.pdf" \
  -F "tags=research,python"   # optional
```

**Success Response:**
```json
{"job_id": "e3ce9f20b6fc", "status": "pending"}
```

### 2. Poll Job Status
```bash
curl -s "$KNOWLEDGE_HUB_BASE_URL/upload/status/$JOB_ID" \
  -H "Authorization: Bearer $KNOWLEDGE_HUB_TOKEN"
```

**Response Fields:**
- `status`: `pending` / `processing` / `done` / `failed`
- `chunks`: number of indexed chunks (when done)
- `error`: error message if failed

**Polling Strategy**: Poll every 2 seconds. Stop when status is `done` or `failed`.

### 3. Report Result
- **Success**: Tell user filename and number of chunks indexed.
- **Failed**: Show the error message directly.

## Reference Python Implementation
```python
import requests
import time

def upload_to_knowledge_hub(file_path: str, base_url: str, token: str, tags: str = None):
    """Upload file and wait for indexing to complete."""
    # Upload
    files = {"file": open(file_path, "rb")}
    data = {"tags": tags} if tags else {}
    headers = {"Authorization": f"Bearer {token}"}
    
    resp = requests.post(f"{base_url}/upload", headers=headers, files=files, data=data)
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    
    # Poll status
    while True:
        status_resp = requests.get(
            f"{base_url}/upload/status/{job_id}", 
            headers=headers
        )
        status = status_resp.json()
        
        if status["status"] in ("done", "failed"):
            return status
        time.sleep(2)
```
