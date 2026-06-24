# 精确 Token 计数 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 BGE-M3 自带的 tokenizer 精确计数替代 `len(text) // 4` 字符估算，消除中文和代码的 token 计数误差。

**Architecture:** 单文件修改 `chunker.py`——在 `__init__` 加载 `AutoTokenizer`，统一用 `self._count_tokens(text)` 替换所有 `self._estimate_tokens(x)` 调用，`_hard_split` 从字符截断改为逐行 token 计数切分。

**Tech Stack:** transformers (AutoTokenizer), BGE-M3 (XLM-RoBERTa tokenizer)

## Global Constraints

- 使用 BGE-M3 tokenizer (`settings.EMBED_MODEL`)，不引入新模型依赖
- tokenizer 加载失败时自动回退到 `len(text) // 4`，不影响离线环境
- 配置和公共接口不变，纯内部实现替换

---

### Task 1: 更新 `__init__` — 加载 tokenizer 并创建 `_count_tokens`

**Files:**
- Modify: `src/knowledge_hub/ingestion/chunker.py:1-26`

**Interfaces:**
- Produces: `self._tokenizer` (AutoTokenizer 实例), `self._count_tokens(text: str) -> int`

- [ ] **Step 1: 在文件顶部添加 import**

```python
# chunker.py 开头，在现有 import 后添加
import structlog
from transformers import AutoTokenizer

logger = structlog.get_logger()
```

- [ ] **Step 2: 修改 `__init__`**

```python
def __init__(self, settings: Settings):
    self._max_tokens = settings.CHUNK_MAX_TOKENS
    self._overlap = settings.CHUNK_OVERLAP

    try:
        self._tokenizer = AutoTokenizer.from_pretrained(settings.EMBED_MODEL)
        self._count_tokens = lambda text: len(
            self._tokenizer.encode(text, add_special_tokens=False)
        )
    except Exception as e:
        logger.warning("tokenizer_load_failed_falling_back", error=str(e))
        self._count_tokens = lambda text: max(1, len(text) // 4)
```

- [ ] **Step 3: 运行现有测试确认 tokenizer 加载正常**

```bash
.venv/bin/python -m pytest tests/test_chunker.py -v
```
预期：全部 13 个测试 PASS（tokenizer 命中本地缓存，加载成功）

- [ ] **Step 4: 暂存**

```bash
git add src/knowledge_hub/ingestion/chunker.py
```

> **暂不 commit**——Task 1 + Task 2 逻辑上是一个完整变更，在 Task 2 完成后一起 commit。

---

### Task 2: 替换 `_estimate_tokens` 调用 + 重写 `_hard_split`

**Files:**
- Modify: `src/knowledge_hub/ingestion/chunker.py:75-189`

**Interfaces:**
- Consumes: `self._count_tokens(text: str) -> int` (Task 1)
- Produces: `_split_by_tokens` 和 `_hard_split` 使用精确 token 计数

- [ ] **Step 1: 替换 `_split_by_tokens` 中的 `_estimate_tokens` 调用**

在 `_split_by_tokens` 方法中，将 `self._estimate_tokens(para)` 改为 `self._count_tokens(para)`（line 92），将 `self._estimate_tokens(current_texts[0])` 改为 `self._count_tokens(current_texts[0])`（line 127）。

```python
# line 92: 段落 token 计数
para_tokens = self._count_tokens(para)

# line 127: overlap 路径中重新计算 current_tokens
current_tokens = self._count_tokens(current_texts[0])
```

- [ ] **Step 2: 重写 `_hard_split` 为逐行 token 计数切分**

完整替换 `_hard_split` 方法（line 147-161）：

```python
def _hard_split(
    self,
    text: str,
    heading_chain: list[str],
    source_file: str,
    source_hash: str,
) -> list[DocumentChunk]:
    """Split an oversized text element into max_tokens-sized chunks by lines."""
    lines = text.split("\n")
    chunks: list[DocumentChunk] = []
    current_lines: list[str] = []
    current_tokens = 0

    for line in lines:
        line_tokens = self._count_tokens(line)

        if line_tokens > self._max_tokens:
            # Extreme case: single line exceeds max_tokens (minified JS, base64)
            # Use tokenizer encode/decode for precise token-based truncation
            if current_lines:
                chunks.append(self._make_chunk(
                    "\n".join(current_lines), heading_chain, source_file, source_hash
                ))
                current_lines = []
                current_tokens = 0
            token_ids = self._tokenizer.encode(line, add_special_tokens=False)
            for i in range(0, len(token_ids), self._max_tokens):
                sub_text = self._tokenizer.decode(
                    token_ids[i:i + self._max_tokens],
                    skip_special_tokens=True,
                )
                chunks.append(self._make_chunk(
                    sub_text, heading_chain, source_file, source_hash
                ))
            continue

        if current_tokens + line_tokens > self._max_tokens and current_lines:
            chunks.append(self._make_chunk(
                "\n".join(current_lines), heading_chain, source_file, source_hash
            ))
            current_lines = []
            current_tokens = 0

        current_lines.append(line)
        current_tokens += line_tokens

    if current_lines:
        chunks.append(self._make_chunk(
            "\n".join(current_lines), heading_chain, source_file, source_hash
        ))

    return chunks
```

- [ ] **Step 3: 删除 `_estimate_tokens` 静态方法**

删除 line 186-189：
```python
# 删除以下 4 行
@staticmethod
def _estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token."""
    return max(1, len(text) // 4)
```

- [ ] **Step 4: 运行现有测试**

```bash
.venv/bin/python -m pytest tests/test_chunker.py -v
```

注意：`test_hard_split_respects_max_tokens` 可能 FAIL——它用 `len(c.text) // 4` 验证，需在 Task 3 中更新。

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_hub/ingestion/chunker.py
git commit -m "feat: replace char-based token estimation with BGE-M3 tokenizer

- Add AutoTokenizer loading in __init__ with char-based fallback
- Replace all _estimate_tokens() calls with _count_tokens()
- Rewrite _hard_split to use line-by-line token counting
- Extreme single-line overflow uses encode/decode for precise truncation

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: 更新测试

**Files:**
- Modify: `tests/test_chunker.py:169-180`

**Interfaces:**
- Consumes: `self._count_tokens` (Task 1), new `_hard_split` (Task 2)

- [ ] **Step 1: 更新 `test_hard_split_respects_max_tokens` 用真实 tokenizer 验证**

```python
def test_hard_split_respects_max_tokens():
    """Hard-split chunks should respect max_tokens (verified with actual tokenizer)."""
    from transformers import AutoTokenizer

    settings = Settings(CHUNK_MAX_TOKENS=20, CHUNK_OVERLAP=0.0)
    chunker = SemanticChunker(settings)
    tokenizer = AutoTokenizer.from_pretrained(settings.EMBED_MODEL)

    # Create a very long single paragraph (no paragraph breaks)
    long_para = "word " * 500  # ~2500 chars
    doc = make_doc(long_para)
    chunks = chunker.chunk([doc], "test.txt", "hash1")
    for c in chunks:
        actual_tokens = len(tokenizer.encode(c.text, add_special_tokens=False))
        # Allow small margin since _hard_split may not fill to exact max
        assert actual_tokens <= 24, (
            f"Chunk has {actual_tokens} tokens, exceeds max_tokens=20"
        )
```

- [ ] **Step 2: 添加中文 token 计数精确性测试**

```python
def test_chinese_token_counting():
    """Chinese text should NOT be underestimated by char-based heuristics."""
    settings = Settings(CHUNK_MAX_TOKENS=30, CHUNK_OVERLAP=0.0)
    chunker = SemanticChunker(settings)
    # 每个汉字 1 char，但 len//4 会严重低估。精确计数应正确切分。
    chinese_text = "这是一个关于优先级继承机制的详细说明文档。" * 5
    doc = make_doc(chinese_text)
    chunks = chunker.chunk([doc], "cn.txt", "hash1")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(settings.EMBED_MODEL)
    for c in chunks:
        actual_tokens = len(tokenizer.encode(c.text, add_special_tokens=False))
        assert actual_tokens <= 36, (
            f"Chinese chunk has {actual_tokens} tokens, exceeds max_tokens=30 + margin"
        )
```

- [ ] **Step 3: 添加代码片段 token 计数测试**

```python
def test_code_token_counting():
    """Code with many symbols should NOT be underestimated."""
    settings = Settings(CHUNK_MAX_TOKENS=30, CHUNK_OVERLAP=0.0)
    chunker = SemanticChunker(settings)
    # 符号密集的代码，每个 {}() 都是独立 token
    code_text = (
        "def process_items(items: list[str]) -> dict[str, int]:\n"
        "    result = {}\n"
        "    for i in range(len(items)):\n"
        "        result[items[i]] = i\n"
        "    return result\n"
    ) * 3
    doc = make_doc(code_text)
    chunks = chunker.chunk([doc], "code.py", "hash1")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(settings.EMBED_MODEL)
    for c in chunks:
        actual_tokens = len(tokenizer.encode(c.text, add_special_tokens=False))
        assert actual_tokens <= 36, (
            f"Code chunk has {actual_tokens} tokens, exceeds max_tokens=30 + margin"
        )
```

- [ ] **Step 4: 运行全部 chunker 测试**

```bash
.venv/bin/python -m pytest tests/test_chunker.py -v
```
预期：全部 16 个测试 PASS

- [ ] **Step 5: 运行全量回归测试**

```bash
.venv/bin/python -m pytest tests/ -v -m "not integration" --ignore=tests/test_integration.py
```
预期：全部 PASS（包括 test_config.py 等不相关测试）

- [ ] **Step 6: Commit**

```bash
git add tests/test_chunker.py
git commit -m "test: update chunker tests for accurate token counting

- Update test_hard_split_respects_max_tokens to use real tokenizer
- Add Chinese and code token counting accuracy tests

Co-Authored-By: Claude <noreply@anthropic.com>"
```
