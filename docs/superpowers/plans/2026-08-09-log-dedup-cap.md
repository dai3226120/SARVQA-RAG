# 日志库同图语义去重 + 容量上限 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** flush 入库时对「同图 + 问题级相似度 ≥ 0.92」的问法变体做语义合并（高分优先），并对日志库设置硬上限淘汰（低分最旧先淘汰），防止日志库随问法变体冗余膨胀。

**Architecture:** 在 `merge_log_records` 精确匹配之上加语义层——`dedup_in_batch`（批内同图合并）→ `find_semantic_duplicates`（复用运行期 ExactVectorIndex 召回候选，问题级余弦判定）→ `merge_log_records(semantic_dups=...)`（同一决策表合并，替换携带旧键）→ `enforce_log_cap`（上限淘汰）。异常时降级为仅精确匹配，不阻断入库。

**Tech Stack:** Python 3.13 / pandas / numpy / faiss（ExactVectorIndex）/ bge-m3 嵌入（model.factory.huggingface_embed_model）/ pytest 9（现有 tests/ 基建）

## Global Constraints

- 设计规格：`docs/superpowers/specs/2026-08-09-log-dedup-cap-design.md`（已批准，提交 df017b9）
- 运行期隶属度命中逻辑（`membership_service.py` / `degree_calculator.py` / 阈值 0.65）**一律不改**
- `upsert_records_to_vector_db` 的"全量重写"现有行为**不重构**（本次只新增删除调用）
- 测试遵循现有模式：`tests/conftest.py` 已全局处理 ortools/pandas 导入顺序与 sys.path；复用 `load_script` / `NoopCollection` / `FakeSliceStore` / `_rec()` 工厂风格
- 语义去重判定必须用**问题级**余弦相似度（文档级向量共享切片内容会假性抬高分数，仅用于候选召回）
- 语义替换的 `replaced_keys` 必须携带**旧行**的 (id, question)（`generate_log_vector_id` 依赖它定位删除）
- Windows 平台；提交信息以 `Co-Authored-By: Claude <noreply@anthropic.com>` 结尾

---

### Task 1: 配置项（chroma.yml + RagConfig + 配置测试）

**Files:**
- Modify: `config/chroma.yml`（retrieval 段之后新增 `dedup` 段）
- Modify: `agent/rag/core/config.py`（RagConfig 新增 3 个字段，`log_collection_name` 之后）
- Create: `tests/test_log_dedup.py`（本任务创建测试文件，Task 2/3 继续往里加测试）

**Interfaces:**
- Produces: `rag_config.dedup_sim_threshold: float`（0.92）/ `rag_config.dedup_pre_filter: float`（0.80）/ `rag_config.dedup_max_log_records: int`（0=自动）。Task 2/3 依赖这三个字段。

- [ ] **Step 1: 写失败测试** — 新建 `tests/test_log_dedup.py`：

```python
"""日志库语义去重 + 容量上限测试（同图去重规格见 docs/superpowers/specs/2026-08-09-log-dedup-cap-design.md）"""
import pandas as pd
import pytest

from rag.core.config import rag_config
from rag.membership.log_manager import (
    LOG_COLUMNS, merge_log_records, dedup_in_batch,
    find_semantic_duplicates, enforce_log_cap,
)


def _rec(id_, q, score, ts="2026-08-07 10:00:00", **kw):
    r = {"id": id_, "question": q, "retrieved_slices": "s1", "retrieved_slices_content": "c",
         "predicted_text": "p", "correct": 1, "correctness_score": score, "timestamp": ts}
    r.update(kw)
    return r


def test_dedup_config_reads_from_yml():
    assert rag_config.dedup_sim_threshold == 0.92
    assert rag_config.dedup_pre_filter == 0.80
    assert rag_config.dedup_max_log_records == 0


def test_dedup_config_pre_filter_below_threshold():
    assert rag_config.dedup_pre_filter <= rag_config.dedup_sim_threshold
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_log_dedup.py -v`
Expected: FAIL，`AttributeError: 'RagConfig' object has no attribute 'dedup_sim_threshold'`

- [ ] **Step 3: 实现** — `config/chroma.yml`，在 `retrieval:` 段之后追加：

```yaml
# 日志库语义去重与容量控制（flush_pending_logs.py 入库时生效）
dedup:
  sim_threshold: 0.92   # 同图问题级余弦相似度≥此值判定为问法变体（实测：存量库 max=0.884，零误合并；勿下调至 0.85-0.88 区间，会误合并同图不同问题）
  pre_filter: 0.80      # 文档级相似度预筛阈值，低于此值的同图候选不做问题级复核（省嵌入）
  max_log_records: 0    # 0 = 自动 = round(1.5 × 切片库条数)；正整数 = 硬上限
```

`agent/rag/core/config.py`，`RagConfig` 中 `log_collection_name` 字段之后新增：

```python
    # ── 日志向量库 ──
    log_collection_name: str = "rag_test_logs"

    # ── 日志库语义去重与容量控制 ──
    dedup_sim_threshold: float = float(chroma_conf.get("dedup", {}).get("sim_threshold", 0.92))
    dedup_pre_filter: float = float(chroma_conf.get("dedup", {}).get("pre_filter", 0.80))
    dedup_max_log_records: int = int(chroma_conf.get("dedup", {}).get("max_log_records", 0))
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_log_dedup.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add config/chroma.yml agent/rag/core/config.py tests/test_log_dedup.py
git commit -m "feat(config): 日志库语义去重与容量上限配置项

- dedup.sim_threshold=0.92 / pre_filter=0.80 / max_log_records=0(自动=1.5×切片库)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: log_manager.py 纯逻辑函数（批内去重 / 库内查重 / 上限淘汰 / merge 语义路径）

**Files:**
- Modify: `agent/rag/membership/log_manager.py`（模块顶部加 `import numpy as np`；`merge_log_records` 扩展；文件末尾新增 3 个函数）
- Test: `tests/test_log_dedup.py`（追加本任务测试）

**Interfaces:**
- Consumes: `record_key(id_, question)`（已存在，`log_manager.py:30`）、`LOG_COLUMNS`、`rag_config.dedup_sim_threshold` 等（Task 1）
- Produces（Task 3 依赖）:
  - `dedup_in_batch(new_records: list[dict], embed_fn, sim_threshold: float = 0.92) -> list[int]` — 批内同图高相似合并，返回被丢弃下标
  - `find_semantic_duplicates(existing_df, new_records, logs_index, embed_fn, sim_threshold=0.92, pre_filter=0.80, candidate_k=20) -> dict[int, int]` — {新记录下标: 现有行下标}
  - `enforce_log_cap(df, max_records: int) -> tuple[pd.DataFrame, list[tuple[str, str]]]` — (裁剪后 df, 被淘汰 (id, question) 键列表)
  - `merge_log_records(existing_df, new_records, semantic_dups: dict[int, int] | None = None)` — 语义路径复用同一决策表；替换时 replaced_keys 用旧行键（向后兼容，现有调用不受影响）

- [ ] **Step 1: 写失败测试** — `tests/test_log_dedup.py` 追加：

```python
# ── 假组件：确定性嵌入 + 假日志索引 ──

class DictEmbed:
    """确定性假嵌入：按文本查表返回归一化向量；未登记文本 → 由哈希派生的随机向量"""
    def __init__(self, vecs=None):
        self._vecs = vecs or {}
        self._cache = {}

    def embed_documents(self, texts):
        import hashlib
        import numpy as np
        out = []
        for t in texts:
            if t not in self._cache:
                if t in self._vecs:
                    v = np.array(self._vecs[t], dtype=np.float32)
                else:
                    h = hashlib.md5(t.encode("utf-8")).digest()
                    v = np.array([b / 255.0 - 0.5 for b in h[:8]], dtype=np.float32)
                v = v / (np.linalg.norm(v) + 1e-9)
                self._cache[t] = v.tolist()
            out.append(self._cache[t])
        return out


class FakeLogIndex:
    """假日志库索引：按查询文本返回预设的 [(metadata, doc_sim)]"""
    def __init__(self, hits_by_q):
        self._hits = hits_by_q

    def search_text(self, q, k, qe=None):
        return self._hits.get(q, [])[:k]


# ── dedup_in_batch：批内同图语义去重 ──

def test_in_batch_dedup_keeps_higher_score():
    embed = DictEmbed({"What is shown?": [1, 0, 0, 0], "What is displayed?": [1, 0, 0, 0],
                       "How many?": [0, 1, 0, 0]})
    recs = [
        _rec("a.png", "What is shown?", 0.9, ts="2026-08-07 10:00:00"),
        _rec("a.png", "What is displayed?", 0.7, ts="2026-08-07 11:00:00"),
        _rec("b.png", "What is shown?", 0.8, ts="2026-08-07 10:00:00"),
    ]
    dropped = dedup_in_batch(recs, embed, sim_threshold=0.92)
    assert dropped == [1]  # a.png 低分变体丢弃；b.png 同问法不同图保留


def test_in_batch_dedup_same_score_keeps_newer():
    embed = DictEmbed({"q-a": [1, 0, 0, 0], "q-b": [1, 0, 0, 0]})
    recs = [
        _rec("a.png", "q-a", 0.8, ts="2026-08-07 09:00:00"),
        _rec("a.png", "q-b", 0.8, ts="2026-08-07 12:00:00"),
    ]
    dropped = dedup_in_batch(recs, embed)
    assert dropped == [0]  # 同分保新：后出现的替换先出现的


def test_in_batch_distinct_questions_not_merged():
    embed = DictEmbed({"How many?": [1, 0, 0, 0], "What color?": [0, 1, 0, 0]})
    recs = [_rec("a.png", "How many?", 0.9), _rec("a.png", "What color?", 0.8)]
    dropped = dedup_in_batch(recs, embed)
    assert dropped == []


def test_in_batch_empty_id_or_question_skipped():
    embed = DictEmbed({"q1": [1, 0, 0, 0], "q2": [1, 0, 0, 0]})
    recs = [
        _rec("", "q1", 0.9),            # 空 id：不参与比较，保留
        _rec("a.png", "", 0.9),         # 空 question：不参与比较，保留
        _rec("a.png", "q2", 0.8),
    ]
    dropped = dedup_in_batch(recs, embed)
    assert dropped == []


# ── find_semantic_duplicates：库内语义查重 ──

def test_semantic_dup_found_same_id_high_sim():
    existing = pd.DataFrame(
        [_rec("a.png", "What is shown?", 0.6, ts="2026-08-07 09:00:00")],
        columns=LOG_COLUMNS,
    )
    new = [_rec("a.png", "What is displayed?", 0.9)]
    embed = DictEmbed({"What is shown?": [1, 0, 0, 0], "What is displayed?": [1, 0, 0, 0]})
    index = FakeLogIndex({
        "What is displayed?": [({"id": "a.png", "question": "What is shown?"}, 0.95)],
    })
    dups = find_semantic_duplicates(existing, new, index, embed)
    assert dups == {0: 0}


def test_semantic_dup_same_id_low_question_sim_not_found():
    existing = pd.DataFrame(
        [_rec("a.png", "What is shown?", 0.6)], columns=LOG_COLUMNS,
    )
    new = [_rec("a.png", "How many buildings?", 0.9)]
    embed = DictEmbed({"What is shown?": [1, 0, 0, 0], "How many buildings?": [0, 1, 0, 0]})
    index = FakeLogIndex({
        "How many buildings?": [({"id": "a.png", "question": "What is shown?"}, 0.95)],
    })
    dups = find_semantic_duplicates(existing, new, index, embed)
    assert dups == {}  # 文档级 0.95 过预筛，但问题级正交 → 不判重复


def test_semantic_dup_different_id_not_found():
    existing = pd.DataFrame(
        [_rec("a.png", "What is shown?", 0.6)], columns=LOG_COLUMNS,
    )
    new = [_rec("b.png", "What is shown?", 0.9)]
    embed = DictEmbed({"What is shown?": [1, 0, 0, 0]})
    index = FakeLogIndex({
        "What is shown?": [({"id": "a.png", "question": "What is shown?"}, 0.99)],
    })
    dups = find_semantic_duplicates(existing, new, index, embed)
    assert dups == {}  # 异图相同问题必须各自保留


def test_semantic_dup_skips_exact_match():
    existing = pd.DataFrame(
        [_rec("a.png", "What is shown?", 0.6)], columns=LOG_COLUMNS,
    )
    new = [_rec("a.png", "What is shown?", 0.9)]
    embed = DictEmbed({"What is shown?": [1, 0, 0, 0]})
    index = FakeLogIndex({
        "What is shown?": [({"id": "a.png", "question": "What is shown?"}, 1.0)],
    })
    dups = find_semantic_duplicates(existing, new, index, embed)
    assert dups == {}  # 精确匹配交给 merge_log_records，语义路径跳过


def test_semantic_dup_below_pre_filter_skipped():
    existing = pd.DataFrame(
        [_rec("a.png", "What is shown?", 0.6)], columns=LOG_COLUMNS,
    )
    new = [_rec("a.png", "What is displayed?", 0.9)]
    embed = DictEmbed({"What is shown?": [1, 0, 0, 0], "What is displayed?": [1, 0, 0, 0]})
    index = FakeLogIndex({
        "What is displayed?": [({"id": "a.png", "question": "What is shown?"}, 0.70)],
    })
    dups = find_semantic_duplicates(existing, new, index, embed, pre_filter=0.80)
    assert dups == {}  # 文档级 0.70 < 预筛 0.80，不浪费嵌入


def test_semantic_dup_index_exception_degrades_to_empty():
    existing = pd.DataFrame(
        [_rec("a.png", "What is shown?", 0.6)], columns=LOG_COLUMNS,
    )
    new = [_rec("a.png", "What is displayed?", 0.9)]

    class BrokenIndex:
        def search_text(self, q, k, qe=None):
            raise RuntimeError("index down")

    dups = find_semantic_duplicates(existing, new, BrokenIndex(), DictEmbed())
    assert dups == {}  # 检索异常 → 该记录跳过语义路径，不阻断入库


# ── merge_log_records：语义路径 ──

def test_merge_semantic_higher_score_replaces_with_old_key():
    df = pd.DataFrame(
        [_rec("a.png", "What is shown?", 0.6, ts="2026-08-07 09:00:00")], columns=LOG_COLUMNS,
    )
    new = [_rec("a.png", "What is displayed?", 0.9, ts="2026-08-07 11:00:00")]
    merged, stats, replaced_keys = merge_log_records(df, new, semantic_dups={0: 0})
    assert stats == {"added": 0, "replaced": 1, "dropped": 0}
    assert replaced_keys == [("a.png", "What is shown?")]  # 旧键：向量删除定位
    assert merged.iloc[0]["question"] == "What is displayed?"
    assert merged.iloc[0]["correctness_score"] == 0.9


def test_merge_semantic_lower_score_dropped():
    df = pd.DataFrame(
        [_rec("a.png", "What is shown?", 0.8)], columns=LOG_COLUMNS,
    )
    new = [_rec("a.png", "What is displayed?", 0.6)]
    merged, stats, replaced_keys = merge_log_records(df, new, semantic_dups={0: 0})
    assert stats == {"added": 0, "replaced": 0, "dropped": 1}
    assert replaced_keys == []
    assert merged.iloc[0]["question"] == "What is shown?"  # 保留旧值


def test_merge_semantic_equal_score_newer_replaces():
    df = pd.DataFrame(
        [_rec("a.png", "What is shown?", 0.8, ts="2026-08-07 09:00:00")], columns=LOG_COLUMNS,
    )
    new = [_rec("a.png", "What is displayed?", 0.8, ts="2026-08-07 12:00:00")]
    merged, stats, replaced_keys = merge_log_records(df, new, semantic_dups={0: 0})
    assert stats == {"added": 0, "replaced": 1, "dropped": 0}
    assert merged.iloc[0]["question"] == "What is displayed?"


def test_merge_semantic_adds_rest_as_before():
    df = pd.DataFrame(
        [_rec("a.png", "What is shown?", 0.6)], columns=LOG_COLUMNS,
    )
    new = [
        _rec("a.png", "What is displayed?", 0.9),   # 语义路径 → 替换
        _rec("b.png", "brand new?", 0.7),           # 精确路径 → 新增
    ]
    merged, stats, replaced_keys = merge_log_records(df, new, semantic_dups={0: 0})
    assert stats == {"added": 1, "replaced": 1, "dropped": 0}
    assert len(merged) == 2
    assert replaced_keys == [("a.png", "What is shown?")]


# ── enforce_log_cap：容量上限淘汰 ──

def test_cap_noop_below_limit():
    df = pd.DataFrame([_rec("a.png", "q1", 0.5), _rec("b.png", "q2", 0.9)], columns=LOG_COLUMNS)
    kept, evicted = enforce_log_cap(df, 10)
    assert len(kept) == 2 and evicted == []


def test_cap_evicts_lowest_score_then_oldest():
    df = pd.DataFrame([
        _rec("a.png", "q1", 0.5, ts="2026-08-07 08:00:00"),
        _rec("b.png", "q2", 0.9, ts="2026-08-07 09:00:00"),
        _rec("c.png", "q3", 0.5, ts="2026-08-07 07:00:00"),  # 同 0.5 但更旧 → 先淘汰
        _rec("d.png", "q4", 0.7, ts="2026-08-07 10:00:00"),
    ], columns=LOG_COLUMNS)
    kept, evicted = enforce_log_cap(df, 2)
    assert evicted == [("c.png", "q3"), ("a.png", "q1")]  # 排序: (score升序, ts升序)
    assert len(kept) == 2
    assert set(kept["id"]) == {"b.png", "d.png"}


def test_cap_zero_or_negative_disables():
    df = pd.DataFrame([_rec("a.png", "q1", 0.5)], columns=LOG_COLUMNS)
    kept, evicted = enforce_log_cap(df, 0)
    assert len(kept) == 1 and evicted == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_log_dedup.py -v`
Expected: FAIL，`ImportError: cannot import name 'dedup_in_batch' from 'rag.membership.log_manager'`

- [ ] **Step 3: 实现** — `agent/rag/membership/log_manager.py`：

模块顶部导入（第 9-10 行 `import os` / `import hashlib` 之间保持原样，在 `import pandas as pd` 之后加）：

```python
import numpy as np
```

`merge_log_records` 整体替换为（`log_manager.py:41-95`）：

```python
def merge_log_records(
    existing_df: pd.DataFrame,
    new_records: list[dict],
    semantic_dups: dict[int, int] | None = None,
):
    """按 图片+问题 查重合并日志记录。

    决策表：
      - 键不存在        → added（追加）
      - 存在且新分数更高 → replaced（整体替换，时间戳用新的）
      - 存在且分数相等   → 保留时间戳更晚的一条（同分保新）
      - 存在且新分数更低 → dropped（保留旧值）

    semantic_dups: 可选语义查重结果 {新记录下标: 现有行下标}——新记录是该现有行的
    问法变体（同图高相似），按同一决策表合并；替换时 replaced_keys 携带**旧行**的
    (id, question)（向量删除定位）。

    Returns:
        (merged_df, {"added": int, "replaced": int, "dropped": int}, replaced_keys)
        replaced_keys: 实际发生替换的旧行 (id, question) 键列表，供向量库定位删除旧向量
    """
    df = existing_df.copy()
    if df.empty:
        df = pd.DataFrame(columns=LOG_COLUMNS)
    for col in LOG_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[LOG_COLUMNS]

    stats = {"added": 0, "replaced": 0, "dropped": 0}
    replaced_keys = []
    key_to_idx = {record_key(r["id"], r["question"]): i for i, r in df.iterrows()}
    semantic_dups = semantic_dups or {}

    for i, rec in enumerate(new_records):
        key = record_key(rec["id"], rec["question"])
        new_score = float(rec.get("correctness_score", 0.0))
        new_ts = str(rec.get("timestamp", ""))

        if i in semantic_dups:
            idx = semantic_dups[i]  # 语义重复：直接定位现有行
        elif key in key_to_idx:
            idx = key_to_idx[key]
        else:
            stats["added"] += 1
            row = pd.DataFrame([{c: rec.get(c, "") for c in LOG_COLUMNS}], columns=LOG_COLUMNS)
            df = pd.concat([df, row], ignore_index=True)
            key_to_idx[key] = len(df) - 1
            continue

        old_score = float(df.at[idx, "correctness_score"] or 0.0)
        old_ts = str(df.at[idx, "timestamp"] or "")

        if new_score > old_score:
            stats["replaced"] += 1
            replaced_keys.append((str(df.at[idx, "id"]), str(df.at[idx, "question"])))
            for c in LOG_COLUMNS:
                df.at[idx, c] = rec.get(c, "")
        elif new_score == old_score and new_ts > old_ts:
            stats["replaced"] += 1
            replaced_keys.append((str(df.at[idx, "id"]), str(df.at[idx, "question"])))
            for c in LOG_COLUMNS:
                df.at[idx, c] = rec.get(c, "")
        else:
            stats["dropped"] += 1

    return df, stats, replaced_keys
```

文件末尾（`_build_log_texts` 之后）追加三个函数：

```python
# ====================== 语义去重与容量控制 ======================


def dedup_in_batch(new_records: list[dict], embed_fn, sim_threshold: float = 0.92) -> list[int]:
    """批内同图高相似合并（纯函数）。

    同图记录两两比较问题级余弦相似度，≥ sim_threshold 视为问法变体：
    保留正确率更高的一条；同分保留时间戳更新的一条；再相同保留先出现的一条。
    空 id / 空 question 的记录不参与比较，直接保留。

    Returns: 被丢弃的记录下标列表（其余记录保留）
    """
    if len(new_records) < 2:
        return []

    groups: dict[str, list[int]] = {}
    for i, rec in enumerate(new_records):
        rid = str(rec.get("id", "")).strip()
        q = str(rec.get("question", "")).strip()
        if rid and q:
            groups.setdefault(rid, []).append(i)

    dropped: set[int] = set()
    for rid, idxs in groups.items():
        if len(idxs) < 2:
            continue
        qs = [str(new_records[i]["question"]).strip() for i in idxs]
        embs = np.array(embed_fn.embed_documents(qs), dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        E = embs / norms

        kept = [0]  # 组内已保留的记录在 idxs 中的位置
        for j in range(1, len(idxs)):
            sims = np.array([float(E[j] @ E[k]) for k in kept])
            if sims.max() < sim_threshold:
                kept.append(j)
                continue
            k = kept[int(np.argmax(sims))]  # 与相似度最高的代表比较
            rec_j, rec_k = new_records[idxs[j]], new_records[idxs[k]]
            j_score = float(rec_j.get("correctness_score", 0.0))
            k_score = float(rec_k.get("correctness_score", 0.0))
            j_ts = str(rec_j.get("timestamp", ""))
            k_ts = str(rec_k.get("timestamp", ""))
            if j_score > k_score or (j_score == k_score and j_ts > k_ts):
                kept.remove(k)
                kept.append(j)
            # 否则丢弃 j（保持现代表）

        dropped.update(idxs[j] for j in range(len(idxs)) if j not in kept)

    return sorted(dropped)


def find_semantic_duplicates(
    existing_df: pd.DataFrame,
    new_records: list[dict],
    logs_index,
    embed_fn,
    sim_threshold: float = 0.92,
    pre_filter: float = 0.80,
    candidate_k: int = 20,
) -> dict[int, int]:
    """库内语义查重：对每条新记录查找 同图片id + 问题级相似度≥阈值 的现有行。

    两阶段：文档级向量检索召回候选（日志向量 = 问题+切片+回答，共享切片内容会抬高
    分数，故仅用于召回）→ 问题级余弦相似度精确判定（避免把同图不同问题误判为重复）。

    Returns: {新记录下标: 现有行下标}。空 id / 空 question / 精确匹配已存在的记录
    不参与语义路径（精确匹配由 merge_log_records 处理）；检索异常的单条记录跳过。
    """
    if existing_df.empty or not new_records:
        return {}

    key_to_idx = {
        record_key(str(r["id"]), str(r["question"])): i
        for i, r in existing_df.iterrows()
    }
    dups: dict[int, int] = {}
    for i, rec in enumerate(new_records):
        rid = str(rec.get("id", "")).strip()
        q = str(rec.get("question", "")).strip()
        if not rid or not q or record_key(rid, q) in key_to_idx:
            continue
        try:
            hits = logs_index.search_text(q, candidate_k)  # [(metadata, doc_sim)]
        except Exception:
            continue
        cands = [
            (m, s) for m, s in hits
            if str(m.get("id", "")) == rid and s >= pre_filter
        ]
        if not cands:
            continue

        qe = np.array(embed_fn.embed_documents([q])[0], dtype=np.float32)
        qe = qe / (np.linalg.norm(qe) + 1e-9)
        cand_qs = [str(m.get("question", "")) for m, _ in cands]
        ce = np.array(embed_fn.embed_documents(cand_qs), dtype=np.float32)
        norms = np.linalg.norm(ce, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        ce = ce / norms
        sims = ce @ qe
        best = int(np.argmax(sims))
        if float(sims[best]) >= sim_threshold:
            m = cands[best][0]
            key = record_key(str(m.get("id", "")), str(m.get("question", "")))
            if key in key_to_idx:
                dups[i] = key_to_idx[key]
    return dups


def enforce_log_cap(
    df: pd.DataFrame, max_records: int
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """容量上限淘汰：超限按 (correctness_score 升序, timestamp 升序) 裁掉尾部。

    max_records <= 0 表示不设上限（原样返回）。

    Returns: (裁剪后 df, 被淘汰 (id, question) 键列表)——被淘汰键用于向量库删除定位
    """
    if max_records <= 0 or len(df) <= max_records:
        return df, []
    work = df.copy()
    work["correctness_score"] = pd.to_numeric(
        work["correctness_score"], errors="coerce"
    ).fillna(0.0)
    work["timestamp"] = work["timestamp"].fillna("")
    work = work.sort_values(
        ["correctness_score", "timestamp"], ascending=[True, True]
    ).reset_index(drop=True)
    evicted = work.iloc[: len(work) - max_records]
    kept = work.iloc[len(work) - max_records:]
    evicted_keys = [(str(r["id"]), str(r["question"])) for _, r in evicted.iterrows()]
    return kept.reset_index(drop=True), evicted_keys
```

- [ ] **Step 4: 运行测试确认通过（含既有回归）**

Run: `python -m pytest tests/test_log_dedup.py tests/test_log_merge.py -v`
Expected: PASS（新测试全过 + 既有 merge 测试不受影响——确认 `semantic_dups` 默认 None 时行为不变）

- [ ] **Step 5: 提交**

```bash
git add agent/rag/membership/log_manager.py tests/test_log_dedup.py
git commit -m "feat(rag): 日志库语义去重与容量上限纯逻辑

- dedup_in_batch: 批内同图高相似合并(高分优先/同分保新)
- find_semantic_duplicates: 库内语义查重, 文档级召回+问题级判定
- enforce_log_cap: (score升序, ts升序)裁尾淘汰, 返回被淘汰键
- merge_log_records 新增 semantic_dups 参数, 替换携带旧行键

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: flush_pending_logs.py 编排（语义去重 → 合并 → 上限 → 向量库同步）

**Files:**
- Modify: `agent/data/flush_pending_logs.py`（`run_flush` 改造 + 模块 docstring 更新）
- Test: `tests/test_flush_script.py`（追加 4 个测试）

**Interfaces:**
- Consumes: Task 1 配置、Task 2 的 `dedup_in_batch` / `find_semantic_duplicates` / `enforce_log_cap` / `merge_log_records` / `generate_log_vector_id`；`rag.core.exact_index.ExactVectorIndex`（已存在）；`model.factory.huggingface_embed_model`（已存在）
- Produces: `run_flush(pending_path, log_manager, slice_store, threshold=50, max_log_records=None, logs_index=None, embed_fn=None)` — 新增 3 个可选参数（测试注入用；默认 None 走生产路径）；返回 stats 增加 `in_batch_dropped` 键

- [ ] **Step 1: 写失败测试** — `tests/test_flush_script.py` 追加（复用其 `_manager` helper 与 conftest 组件，`DictEmbed` / `FakeLogIndex` 从 `tests.test_log_dedup` 导入）：

```python
from tests.test_log_dedup import DictEmbed, FakeLogIndex


def _flush_with_semantic(tmp_path, monkeypatch, pending_rows, library_rows,
                         index_hits=None, max_log_records=0, threshold=2):
    """构造 假嵌入+假索引 的 run_flush 环境；返回 (result, manager, pending_path)"""
    flush = load_script("flush_pending_logs")
    m = _manager(tmp_path, monkeypatch)
    m.log_df = pd.DataFrame(library_rows, columns=LOG_COLUMNS)
    p = str(tmp_path / "pending.csv")
    for r in pending_rows:
        append_pending_record(r, pending_path=p)
    monkeypatch.setattr(flush.rag_config, "dedup_max_log_records", max_log_records)
    embed = DictEmbed({"What is shown?": [1, 0, 0, 0], "What is displayed?": [1, 0, 0, 0],
                       "How many?": [0, 1, 0, 0], "brand new?": [0, 0, 1, 0]})
    index = FakeLogIndex(index_hits or {})
    result = flush.run_flush(p, m, FakeSliceStore(), threshold=threshold,
                             logs_index=index, embed_fn=embed)
    return result, m, p


def test_flush_semantic_replaces_old_vector_deleted(tmp_path, monkeypatch):
    result, m, p = _flush_with_semantic(
        tmp_path, monkeypatch,
        pending_rows=[_pending_rec("a.png", "What is displayed?", 0.9)],
        library_rows=[{
            "id": "a.png", "question": "What is shown?", "retrieved_slices": "s0",
            "retrieved_slices_content": "old", "predicted_text": "old", "correct": 1,
            "correctness_score": 0.6, "timestamp": "2026-08-07 08:00:00",
        }],
        index_hits={"What is displayed?": [({"id": "a.png", "question": "What is shown?"}, 0.95)]},
    )
    assert result["flushed"] is True
    assert result["replaced"] == 1 and result["added"] == 0
    # 旧向量按旧键删除（generate_log_vector_id = md5(旧问题) 派生，见 log_manager.py:35-38）
    from rag.membership.log_manager import generate_log_vector_id
    assert generate_log_vector_id("a.png", "What is shown?") in m._logs_collection.deleted
    df = pd.read_csv(tmp_path / "logs.csv")
    assert len(df) == 1 and df.iloc[0]["question"] == "What is displayed?"


def test_flush_semantic_lower_score_dropped_keeps_library(tmp_path, monkeypatch):
    result, m, p = _flush_with_semantic(
        tmp_path, monkeypatch,
        pending_rows=[_pending_rec("a.png", "What is displayed?", 0.4)],
        library_rows=[{
            "id": "a.png", "question": "What is shown?", "retrieved_slices": "s0",
            "retrieved_slices_content": "old", "predicted_text": "old", "correct": 1,
            "correctness_score": 0.8, "timestamp": "2026-08-07 08:00:00",
        }],
        index_hits={"What is displayed?": [({"id": "a.png", "question": "What is shown?"}, 0.95)]},
    )
    assert result["dropped"] == 1 and result["replaced"] == 0
    df = pd.read_csv(tmp_path / "logs.csv")
    assert len(df) == 1 and df.iloc[0]["question"] == "What is shown?"


def test_flush_in_batch_dedup_merges_before_library(tmp_path, monkeypatch):
    # 库为空；批内两条同图 paraphrase（高分 0.9 / 低分 0.7）→ 合并为 1 条新增
    result, m, p = _flush_with_semantic(
        tmp_path, monkeypatch,
        pending_rows=[
            _pending_rec("a.png", "What is shown?", 0.9),
            _pending_rec("a.png", "What is displayed?", 0.7),
        ],
        library_rows=[],
        threshold=2,
    )
    assert result["flushed"] is True
    assert result["added"] == 1
    assert result["in_batch_dropped"] == 1
    df = pd.read_csv(tmp_path / "logs.csv")
    assert len(df) == 1 and float(df.iloc[0]["correctness_score"]) == 0.9


def test_flush_cap_evicts_lowest_and_deletes_vector(tmp_path, monkeypatch):
    from rag.membership.log_manager import generate_log_vector_id
    result, m, p = _flush_with_semantic(
        tmp_path, monkeypatch,
        pending_rows=[_pending_rec("b.png", "How many?", 0.9)],
        library_rows=[
            {"id": "a.png", "question": "What is shown?", "retrieved_slices": "s0",
             "retrieved_slices_content": "old", "predicted_text": "old", "correct": 1,
             "correctness_score": 0.5, "timestamp": "2026-08-07 08:00:00"},
            {"id": "c.png", "question": "brand new?", "retrieved_slices": "s0",
             "retrieved_slices_content": "old", "predicted_text": "old", "correct": 1,
             "correctness_score": 0.7, "timestamp": "2026-08-07 09:00:00"},
        ],
        max_log_records=2,
        threshold=1,
    )
    assert result["flushed"] is True
    df = pd.read_csv(tmp_path / "logs.csv")
    assert len(df) == 2  # 3 条 → 上限 2
    assert "a.png" not in set(df["id"])  # 0.5 最低被淘汰
    assert generate_log_vector_id("a.png", "What is shown?") in m._logs_collection.deleted
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_flush_script.py -v`
Expected: FAIL（旧 2 个测试通过；新测试因 run_flush 尚无 `logs_index`/`embed_fn` 参数或行为未实现而失败）

- [ ] **Step 3: 实现** — `agent/data/flush_pending_logs.py`：

模块 docstring（第 1-9 行）更新为：

```python
"""
隶属度历史日志批量入库脚本（手动运行）
检测暂存区（agent/data/pending_records.csv）待入库记录条数，
超过阈值后按 图片+问题 查重合并（新加/高分替换）并批量写入日志库与向量库，
成功后备份并清空暂存区。

语义去重（同图问题级相似度 ≥ dedup.sim_threshold 视为问法变体）：
  1. 批内合并：待入库记录之间同图高相似 → 高分优先合并
  2. 库内查重：对每条新记录召回日志库同图候选（文档级向量）→ 问题级余弦复核
  3. 决策表：新分更高 → 替换（旧向量删除后重写）；同分保新；更低 → 丢弃
容量上限（dedup.max_log_records，0=自动=1.5×切片库条数）：
  超限按 (正确率升序, 时间戳升序) 淘汰，被淘汰记录同时删除向量。
语义路径异常自动降级为仅精确匹配，不阻断入库。

用法:
    python agent/data/flush_pending_logs.py                 # 默认阈值 50
    python agent/data/flush_pending_logs.py --threshold 20
"""
```

`run_flush` 整体替换（`flush_pending_logs.py:37-84`）：

```python
def run_flush(pending_path, log_manager, slice_store, threshold=50,
              max_log_records=None, logs_index=None, embed_fn=None):
    """执行批量入库。返回统计 dict；pending 条数 < threshold 时不入库、不清空。

    max_log_records: 覆盖配置（测试注入）；None → rag_config.dedup_max_log_records（0=自动）
    logs_index / embed_fn: 语义去重组件注入（测试用）；None → 生产组件
    """
    pending_count = count_pending_records(pending_path)
    if pending_count < threshold:
        print(f"⏭ 暂存区记录 {pending_count} 条 < 阈值 {threshold}，跳过入库")
        return {"pending": pending_count, "added": 0, "replaced": 0,
                "dropped": 0, "flushed": False}

    df = pd.read_csv(pending_path)
    records = df.to_dict("records")

    embed_fn = embed_fn or huggingface_embed_model
    sim_threshold = rag_config.dedup_sim_threshold
    pre_filter = rag_config.dedup_pre_filter

    # 语义去重（异常降级为仅精确匹配，不阻断入库）
    in_batch_dropped = 0
    semantic_dups = {}
    try:
        # ① 批内同图高相似合并（先于切片内容补取，少做无用 get_by_ids）
        dropped_idx = dedup_in_batch(records, embed_fn, sim_threshold)
        records = [r for i, r in enumerate(records) if i not in set(dropped_idx)]
        in_batch_dropped = len(dropped_idx)

        if records:
            # ② 库内语义查重（文档级召回候选 + 问题级判定）
            if logs_index is None:
                logs_index = ExactVectorIndex(
                    collection=log_manager.logs_collection._collection,
                    persist_directory=rag_config.persist_directory,
                    collection_name=rag_config.log_collection_name,
                    embedding_fn=embed_fn,
                )
            semantic_dups = find_semantic_duplicates(
                log_manager.log_df, records, logs_index, embed_fn,
                sim_threshold, pre_filter,
            )
    except Exception as e:
        logger.warning("[flush] 语义去重失败，降级为仅精确匹配: %s", e)

    # 补切片内容
    for rec in records:
        slice_ids = str(rec.get("retrieved_slices", "")).split("|")
        rec["retrieved_slices_content"] = fetch_slices_content(slice_store, slice_ids)
        rec["correct"] = int(float(rec.get("correct", 0))) if not pd.isna(rec.get("correct")) else 0
        rec["correctness_score"] = float(rec.get("correctness_score", 0.0))

    # ③ 查重合并（精确 + 语义；replaced_keys 为实际被替换的旧行键）
    merged_df, stats, replaced_keys = merge_log_records(
        log_manager.log_df, records, semantic_dups=semantic_dups
    )
    stats["in_batch_dropped"] = in_batch_dropped

    # ④ 容量上限淘汰
    if max_log_records is None:
        max_log_records = rag_config.dedup_max_log_records
        if max_log_records == 0:
            max_log_records = round(1.5 * slice_store.count())
    merged_df, evicted_keys = enforce_log_cap(merged_df, max_log_records)

    # 写 CSV
    os.makedirs(os.path.dirname(rag_config.log_path), exist_ok=True)
    merged_df.to_csv(rag_config.log_path, index=False, encoding="utf-8")
    log_manager.log_df = merged_df

    # 向量库：先删被替换旧向量与被淘汰向量，再批量 add 全量新记录
    if evicted_keys:
        evict_ids = [generate_log_vector_id(i, q) for i, q in evicted_keys]
        try:
            log_manager.logs_collection.delete(ids=evict_ids)
            logger.info("[flush] 容量上限淘汰 %d 条并删除对应向量", len(evict_ids))
        except Exception as e:
            logger.error("[flush] 淘汰记录向量删除失败: %s", e)

    upsert_result = log_manager.upsert_records_to_vector_db(
        merged_df.to_dict("records"), replaced_keys
    )

    # 设计条款：向量库失败 → 保留 pending 不清空（CSV 已更新属可接受，下次重跑幂等）
    if upsert_result.get("added", 0) != len(merged_df):
        logger.warning(
            "[flush] 向量库 upsert 未全量成功（added=%s/%s），保留 pending 供下次重试",
            upsert_result.get("added", 0), len(merged_df),
        )
        print(f"⚠️ 向量库写入失败（{upsert_result.get('added', 0)}/{len(merged_df)}），"
              f"pending 保留待重试（CSV 已更新）")
        return {"pending": pending_count, **stats, "flushed": False}

    # 备份并清空暂存区
    backup = clear_pending_records(pending_path)
    logger.info("暂存区已清空，备份: %s", backup)

    print(f"✅ 入库完成: 新增 {stats['added']} | 替换 {stats['replaced']} | "
          f"丢弃 {stats['dropped']} | 批内合并 {stats['in_batch_dropped']} | "
          f"淘汰 {len(evicted_keys)} | 暂存 {pending_count} 条已清空")
    return {"pending": pending_count, **stats, "flushed": True}
```

导入区更新（`flush_pending_logs.py:28-33`）：

```python
from rag.core.config import rag_config
from rag.core.exact_index import ExactVectorIndex
from rag.membership.log_manager import (
    LOG_COLUMNS, LogManager, merge_log_records, fetch_slices_content,
    generate_log_vector_id, dedup_in_batch, find_semantic_duplicates,
    enforce_log_cap,
)
from rag.membership.staging import count_pending_records, clear_pending_records
from rag.stores.slice_store import SliceStore
from utils.logger_handler import logger
```

- [ ] **Step 4: 运行测试确认通过（含既有回归）**

Run: `python -m pytest tests/test_flush_script.py tests/test_log_dedup.py tests/test_log_merge.py -v`
Expected: PASS（新 4 个 + 既有全部）

- [ ] **Step 5: 提交**

```bash
git add agent/data/flush_pending_logs.py tests/test_flush_script.py
git commit -m "feat(rag): flush 入库接入同图语义去重与容量上限

- run_flush: 批内合并→库内语义查重→合并→上限淘汰→向量同步
- 语义路径异常降级为仅精确匹配，不阻断入库
- 被替换/被淘汰记录的向量按旧键删除

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: 全量回归 + 真库冒烟（手动验证）

**Files:**
- Modify: 无（仅验证与确认）

- [ ] **Step 1: 全量测试回归**

Run: `python -m pytest tests/ -v`
Expected: PASS（全部既有 + 新增测试；若有个别与本次改动无关的既有失败，记录输出并说明，不得静默绕过）

- [ ] **Step 2: 真库冒烟（手动，可选但建议）** — 用临时路径跑一次带 paraphrase 的真实 flush：

```bash
# 备份真实日志库后再操作（保护数据）
cp agent/data/rag_feedback_logs.csv agent/data/rag_feedback_logs_backup_pre_dedup.csv
```

用如下临时脚本（`tmp_flush_smoke.py`，用完删除）验证真实模型 + 真实 ExactVectorIndex + 真实 Chroma 的语义去重链路：

```python
import k_means_constrained  # noqa: F401  必须先于 pandas（WinError 127 规避）
import os, sys
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("agent"))

import pandas as pd
from rag.core.config import rag_config
from rag.membership.log_manager import LogManager, generate_log_vector_id
from rag.membership.staging import append_pending_record
from rag.stores.slice_store import SliceStore
from model.factory import huggingface_embed_model
import agent.data.flush_pending_logs as flush

# 隔离：临时 data_path（log_name 派生自 data_path）
tmp = os.path.abspath("agent/data/tmp_dedup_smoke")
os.makedirs(tmp, exist_ok=True)
rag_config.data_path = tmp
rag_config.persist_directory = os.path.join(tmp, "chroma")
rag_config.md5_log_store_path = os.path.join(tmp, "md5.txt")
rag_config.dedup_max_log_records = 0

lm = LogManager(clear_vector_db_only=True)   # 清空向量库重建（CSV 为空 → 空库）
p = os.path.join(tmp, "pending.csv")
append_pending_record({"id": "smoke_img.png", "question": "What vehicles are visible?",
                       "retrieved_slices": "s1", "predicted_text": "Cars and trucks.",
                       "correct": 1, "correctness_score": 0.9,
                       "timestamp": "2026-08-09 10:00:00"}, pending_path=p)
append_pending_record({"id": "smoke_img.png", "question": "Which vehicles can be seen?",
                       "retrieved_slices": "s1", "predicted_text": "Cars and trucks.",
                       "correct": 1, "correctness_score": 0.7,
                       "timestamp": "2026-08-09 10:01:00"}, pending_path=p)
r = flush.run_flush(p, lm, SliceStore(), threshold=1)
print("flush result:", r)
df = pd.read_csv(os.path.join(tmp, "rag_feedback_logs.csv"))
assert len(df) == 1, f"批内去重应合并为 1 条，实际 {len(df)}"
assert df.iloc[0]["question"] == "What vehicles are visible?"
print("SMOKE OK: 批内 paraphrase 已合并，保留高分代表")

# 验证向量库可检索到该记录
from rag.core.exact_index import ExactVectorIndex
idx = ExactVectorIndex(lm.logs_collection._collection, rag_config.persist_directory,
                       rag_config.log_collection_name, embedding_fn=huggingface_embed_model)
hits = idx.search_text("What vehicles are visible?", 3)
print("vector hits:", [(m.get("question"), round(s, 4)) for m, s in hits])
assert hits and str(hits[0][0].get("question", "")).startswith("What vehicles")
print("SMOKE OK: 合并后记录可被精确检索命中")
```

Run: `python tmp_flush_smoke.py`
Expected: 两条输出 `SMOKE OK`；flush result 显示 `新增 1 | 批内合并 1`；真实模型下 paraphrase 相似度 ≥ 0.92（若该对 paraphrase 相似度低于 0.92，输出实际相似度，将 `dedup.sim_threshold` 按实测下调至安全值并重跑）

验证后清理：

```bash
rm -rf agent/data/tmp_dedup_smoke tmp_flush_smoke.py
```

- [ ] **Step 3: 提交（若有阈值调整）**

仅当 Step 2 调整了阈值时执行：

```bash
git add config/chroma.yml
git commit -m "tune(rag): 按真库冒烟实测调整 dedup.sim_threshold

Co-Authored-By: Claude <noreply@anthropic.com>"
```

- [ ] **Step 4: 收尾检查** — 确认 `git status` 只包含预期文件（4 个本次改动 + 你工作区原有 6 个未提交改动不被误动）
