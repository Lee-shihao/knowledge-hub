# 精确 Token 计数

## 问题

`SemanticChunker._estimate_tokens` 使用 `len(text) // 4` 估算 token 数，基于英文 ~4 chars/token 的经验规律。中文和代码的 chars/token 比率差异巨大，导致 chunk 实际 token 数严重超出 `CHUNK_MAX_TOKENS` 限制，送给 BGE-M3 时可能被截断，末尾内容丢失。

### 各语言 chars/token 差异

| 内容类型 | 示例 | chars | 估算 token | 实际 token | 误差 |
|---------|------|-------|------------|-----------|------|
| 英文 | "The quick brown fox" | 19 | 4 | 4 | 0% |
| 中文 | "快速的棕色狐狸跳过了懒狗" | 12 | 3 | ~10 | +233% |
| Python 代码 | `for i in range(len(items)):` | 29 | 7 | ~12 | +71% |

`{`, `(`, `)`, `:` 等符号每个都是独立 token，实际 token 数约是估算的 1.5-2 倍。

### 影响范围

- `_estimate_tokens`（line 187）— 段落 token 计数
- `_hard_split`（line 156）— 超大段落硬切分，用 `max_tokens * 4` 反算字符数

## 方案

使用 BGE-M3 自带的 tokenizer（`AutoTokenizer.from_pretrained`）进行精确 token 计数，替代字符估算。

### 修改文件

`src/knowledge_hub/ingestion/chunker.py`

### 改动点

#### `__init__`

```python
from transformers import AutoTokenizer

class SemanticChunker:
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

- `transformers` 已是项目显式依赖（`transformers>=4.40,<5.0`），无需修改 `pyproject.toml`
- BGE-M3 的 tokenizer 是 XLM-RoBERTa 分词器（sentencepiece.bpe.model ~5MB + tokenizer.json ~16MB，合计约 20MB）。FlagEmbeddingEmbedder 在嵌入阶段已通过 `_ensure_model_cached` 下载完整模型到 HuggingFace hub 缓存目录，chunker 加载 tokenizer 时直接命中本地缓存，**不会触发额外下载**
- 失败时回退到 `len(text) // 4`，确保离线/未缓存环境仍可工作

#### `_estimate_tokens` → `_count_tokens`

删除 `@staticmethod` 的 `_estimate_tokens`，统一使用 `self._count_tokens(text)`。

`_split_by_tokens` 和 `_hard_split` 内部所有 `self._estimate_tokens(x)` 调用改为 `self._count_tokens(x)`。包括：

- `_split_by_tokens` 段落累加循环中的 `para_tokens` 和 `current_tokens` 计算
- `_split_by_tokens` overlap 路径中的 `current_tokens` 重新计算（line 127）
- `_hard_split` 逐行累加中的 `line_tokens` 计算

#### `_hard_split` — 逐行 token 计数切分

```python
def _hard_split(self, text, heading_chain, source_file, source_hash):
    lines = text.split("\n")
    chunks = []
    current_lines = []
    current_tokens = 0

    for line in lines:
        line_tokens = self._count_tokens(line)

        if line_tokens > self._max_tokens:
            # 极端情况：单行超限（minified JS、base64 等）
            # 用 tokenizer encode/decode 按 token 精确截断，与主路径一致
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

与 `_split_by_tokens` 的段落累加逻辑对称，切分单位从段落变为行。

### 性能

- `AutoTokenizer.from_pretrained` 首次加载 ~100ms（冷），后续命中缓存 ~1ms（热）
- `self._count_tokens` 每次调用比 `len() // 4` 慢，但 chunking 阶段没有 embedding 计算，完全可以接受
- 对摄入整体吞吐影响可忽略不计

### 向后兼容

- 配置和接口不变，纯内部实现替换
- 已有 chunk 的 `chunk_id` 不变（ID 基于文本内容 + source_file，与 token 计数方式无关）
- 唯一的副作用是重新摄入时 chunk 边界可能偏移（因为切分位置更精确），产生新的 chunk ID → 旧 chunk 被清理，新 chunk 入库
