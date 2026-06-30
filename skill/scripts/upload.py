#!/usr/bin/env python3
"""
Upload a file to the knowledge-hub knowledge base and wait for indexing to finish.

Reads configuration from environment variables:
  KNOWLEDGE_HUB_BASE_URL  e.g. http://192.168.30.125:8766
  KNOWLEDGE_HUB_TOKEN     bearer token

Usage:
  python upload.py <file_path> [--tags tag1,tag2] [--timeout 60]

Exit codes:
  0  success (indexed)
  1  upload request failed (bad format / auth / network)
  2  indexing failed server-side
  3  timed out waiting for indexing to finish
  4  missing configuration (env vars not set)
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
import mimetypes
import uuid


def build_multipart(file_path: str, tags: str | None):
    boundary = uuid.uuid4().hex
    filename = os.path.basename(file_path)
    mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    with open(file_path, "rb") as f:
        file_data = f.read()

    parts = []
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
    )
    parts.append(f"Content-Type: {mime_type}\r\n\r\n".encode())
    parts.append(file_data)
    parts.append(b"\r\n")

    if tags:
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(b'Content-Disposition: form-data; name="tags"\r\n\r\n')
        parts.append(tags.encode())
        parts.append(b"\r\n")

    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def upload_file(base_url: str, token: str, file_path: str, tags: str | None):
    body, content_type = build_multipart(file_path, tags)
    req = urllib.request.Request(
        f"{base_url}/upload",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": content_type,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode(errors="replace")
        print(f"Upload failed: HTTP {e.code} — {err_body}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Upload failed: network error — {e.reason}", file=sys.stderr)
        sys.exit(1)


def poll_status(base_url: str, token: str, job_id: str, timeout: int):
    req_url = f"{base_url}/upload/status/{job_id}"
    deadline = time.time() + timeout

    while time.time() < deadline:
        req = urllib.request.Request(
            req_url, headers={"Authorization": f"Bearer {token}"}
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = json.loads(resp.read().decode())
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            print(f"Status check failed: {e}", file=sys.stderr)
            sys.exit(1)

        if status["status"] == "done":
            return status
        if status["status"] == "failed":
            print(f"Indexing failed: {status.get('error')}", file=sys.stderr)
            print(json.dumps(status, ensure_ascii=False, indent=2))
            sys.exit(2)

        time.sleep(2)

    print(f"Timed out after {timeout}s waiting for job {job_id} to finish", file=sys.stderr)
    sys.exit(3)


def main():
    parser = argparse.ArgumentParser(description="Upload a file to knowledge-hub")
    parser.add_argument("file_path", help="Path to the file to upload")
    parser.add_argument("--tags", default=None, help="Comma-separated tags")
    parser.add_argument(
        "--timeout", type=int, default=60, help="Max seconds to wait for indexing"
    )
    args = parser.parse_args()

    base_url = os.environ.get("KNOWLEDGE_HUB_BASE_URL")
    token = os.environ.get("KNOWLEDGE_HUB_TOKEN")

    if not base_url or not token:
        print(
            "Missing config: set KNOWLEDGE_HUB_BASE_URL and KNOWLEDGE_HUB_TOKEN",
            file=sys.stderr,
        )
        sys.exit(4)

    if not os.path.isfile(args.file_path):
        print(f"File not found: {args.file_path}", file=sys.stderr)
        sys.exit(1)

    base_url = base_url.rstrip("/")

    upload_resp = upload_file(base_url, token, args.file_path, args.tags)
    job_id = upload_resp["job_id"]
    print(f"Uploaded. job_id={job_id}, status={upload_resp['status']}")

    final_status = poll_status(base_url, token, job_id, args.timeout)
    print(json.dumps(final_status, ensure_ascii=False, indent=2))
    print(
        f"\n✅ Indexed {final_status.get('chunks')} chunk(s) "
        f"from {final_status.get('filename')}"
    )


if __name__ == "__main__":
    main()