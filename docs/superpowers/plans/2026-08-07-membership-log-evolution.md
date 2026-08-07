# 隶属度历史日志形成机制改造 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 改造隶属度历史日志：初始化只由预测正确的会话记录形成（test.csv 重跑 + LLM 判断）、运行期记录按 图片+问题 查重（高分替换）、综合向量含回答、手动批量入库脚本、两种召回方式效率对比。

**Architecture:** 改造现有 `LogManager`（schema 加 `correct`/`retrieved_slices_content` 列、向量文本加 Answer 段、去重改为 图片+问题 高分替换、确定性向量 id 支持替换）；新增三个 `agent/data/` 下的手动脚本（初始化 / 批量入库 / 对比实验）与一个线程安全暂存写入函数；`MembershipCalculator`/`MembershipHybridService` 等调用方接口零改动。

**Tech Stack:** Python 3.10+、pandas、langchain-chroma、pytest、HuggingFace 嵌入模型

## Global Constraints

- 查重键 = `id`（图片文件名）+ `question` 联合；向量 id 格式 `log_<id>_<question_md5前8位>`（确定性）
- 替换依据 = `correctness_score` 连续分数；新分数**更高**才替换；相等保留时间戳更晚（同分保新）
- 向量文本三段式：`Question: {question}` + `Retrieved Slices: {切片内容}` + `Answer: {predicted_text}`；每切片截断 200 字符、总长截断 1500 字符
- 日志 CSV 列序固定：`id, question, retrieved_slices, retrieved_slices_content, predicted_text, correct, correctness_score, timestamp`
- 初始化只保留 `correct == 1` 的记录；运行期 pending 记录 correct 0/1 均可能（由在线系统提供）
- `correct` 为 0/1（int），`correctness_score` 为 float
- 所有测试不调用真实 VLM/LLM API，不读写 `agent/data/` 真实文件；路径全部注入临时目录
- 测试需 `sys.path` 含 `agent/`（`agent/rag/**` 内部以 `from rag.xxx import` 导入）

---

### Task 1: 测试基础设施 + 查重/替换纯逻辑（决策表）+ 确定性 id

**Files:**
- Create: `tests/__init__.py`、`tests/conftest.py`
- Modify: `agent/rag/membership/log_manager.py`（新增纯函数，不改现有行为）
- Test: `tests/test_log_merge.py`（新建）

**Interfaces:**
- Produces:
  - `LOG_COLUMNS: list[str]` — 8 列固定列序
  - `merge_log_records(existing_df: pd.DataFrame, new_records: list[dict]) -> tuple[pd.DataFrame, dict, list[tuple[str, str]]]`
    返回 `(merged_df, {"added": int, "replaced": int, "dropped": int}, replaced_keys)`
    - `replaced_keys`：实际发生替换的 (id, question) 键列表，供向量库定位删除旧向量
    - 决策表：键 `id+question` 不存在 → added；存在且新分数更高 → replaced（行整体替换，时间戳用新的）；存在且分数不高 → dropped；分数相等 → 保留时间戳更晚
  - `generate_log_vector_id(base_id: str, question: str) -> str` — `log_<base_id>_<md5(question)前8位>`
  - `record_key(id: str, question: str) -> str` — `f"{id}__{question}"`

- [ ] **Step 1: 创建测试基础设施（conftest 提供 agent/ sys.path bootstrap 与脚本加载器）**

```python
# tests/__init__.py
# 空文件，标记 tests 为包，支持 from tests.xxx import 共享辅助类
```

```python
# tests/conftest.py
"""全局测试配置：agent/ 加入 sys.path；提供共享 fake 类与脚本加载器"""
import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_AGENT = _ROOT / "agent"
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class NoopCollection:
    """不落盘的 Chroma fake：记录 add/delete，get 返回空"""
    def __init__(self):
        self.calls = []
        self.deleted = []

    def add_texts(self, texts=None, metadatas=None, ids=None):
        self.calls.append({"texts": texts, "metadatas": metadatas, "ids": ids})

    def delete(self, ids=None):
        self.deleted.extend(ids)

    def get(self, **kwargs):
        return {"ids": [], "documents": [], "metadatas": []}


class FakeSliceStore:
    """切片库 fake：get_by_ids 按 id 生成假内容"""
    def get_by_ids(self, ids):
        return {
            "ids": ids,
            "documents": [f"content_of_{i}" for i in ids],
            "metadatas": [{} for _ in ids],
        }


def load_script(name: str):
    """用 importlib 加载 agent/data/ 下的脚本（不依赖 agent.data 包结构）"""
    path = _ROOT / "agent" / "data" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
```

- [ ] **Step 2: 写失败测试**

```python
# tests/test_log_merge.py
import pandas as pd
import pytest

from rag.membership.log_manager import (
    LOG_COLUMNS,
    merge_log_records,
    generate_log_vector_id,
    record_key,
)


def _rec(id_, q, score, ts="2026-08-07 10:00:00", **kw):
    r = {"id": id_, "question": q, "retrieved_slices": "s1", "retrieved_slices_content": "c",
         "predicted_text": "p", "correct": 1, "correctness_score": score, "timestamp": ts}
    r.update(kw)
    return r


def test_columns_fixed_order():
    assert LOG_COLUMNS == ["id", "question", "retrieved_slices", "retrieved_slices_content",
                           "predicted_text", "correct", "correctness_score", "timestamp"]


def test_new_record_added():
    df = pd.DataFrame(columns=LOG_COLUMNS)
    merged, stats, replaced_keys = merge_log_records(df, [_rec("img1.png", "q1", 0.8)])
    assert stats == {"added": 1, "replaced": 0, "dropped": 0}
    assert replaced_keys == []
    assert len(merged) == 1


def test_higher_score_replaces():
    df = pd.DataFrame([_rec("img1.png", "q1", 0.5, ts="2026-08-07 09:00:00")], columns=LOG_COLUMNS)
    merged, stats, replaced_keys = merge_log_records(
        df, [_rec("img1.png", "q1", 0.9, ts="2026-08-07 11:00:00")]
    )
    assert stats == {"added": 0, "replaced": 1, "dropped": 0}
    assert replaced_keys == [("img1.png", "q1")]
    assert merged.iloc[0]["correctness_score"] == 0.9
    assert merged.iloc[0]["timestamp"] == "2026-08-07 11:00:00"


def test_lower_score_dropped():
    df = pd.DataFrame([_rec("img1.png", "q1", 0.8, ts="2026-08-07 09:00:00")], columns=LOG_COLUMNS)
    merged, stats, replaced_keys = merge_log_records(
        df, [_rec("img1.png", "q1", 0.3, ts="2026-08-07 11:00:00")]
    )
    assert stats == {"added": 0, "replaced": 0, "dropped": 1}
    assert replaced_keys == []
    assert len(merged) == 1
    assert merged.iloc[0]["correctness_score"] == 0.8  # 旧值保留


def test_equal_score_keeps_newer_timestamp():
    df = pd.DataFrame([_rec("img1.png", "q1", 0.5, ts="2026-08-07 09:00:00")], columns=LOG_COLUMNS)
    merged, stats, replaced_keys = merge_log_records(
        df, [_rec("img1.png", "q1", 0.5, ts="2026-08-07 12:00:00")]
    )
    assert stats == {"added": 0, "replaced": 1, "dropped": 0}
    assert replaced_keys == [("img1.png", "q1")]
    assert merged.iloc[0]["timestamp"] == "2026-08-07 12:00:00"


def test_same_image_different_question_both_kept():
    df = pd.DataFrame(columns=LOG_COLUMNS)
    merged, stats, _ = merge_log_records(
        df, [_rec("img1.png", "q1", 0.5), _rec("img1.png", "q2", 0.6)]
    )
    assert stats["added"] == 2
    assert len(merged) == 2


def test_record_key_and_vector_id():
    assert record_key("a.png", "你好") == "a.png__你好"
    vid = generate_log_vector_id("a.png", "你好")
    assert vid.startswith("log_a.png_")
    assert generate_log_vector_id("a.png", "你好") == vid  # 确定性
```

- [ ] **Step 3: 运行确认失败**

Run: `python -m pytest tests/test_log_merge.py -v`
Expected: FAIL（ImportError: cannot import name 'merge_log_records'）

- [ ] **Step 4: 实现纯函数**

在 `agent/rag/membership/log_manager.py` 顶部（import 之后）追加：

```python
# ====================== 新 schema 常量与查重/替换纯逻辑 ======================
LOG_COLUMNS = [
    "id", "question", "retrieved_slices", "retrieved_slices_content",
    "predicted_text", "correct", "correctness_score", "timestamp",
]


def record_key(id_: str, question: str) -> str:
    """图片+问题 联合查重键"""
    return f"{id_}__{question}"


def generate_log_vector_id(base_id: str, question: str) -> str:
    """确定性向量库 id：同一 (图片, 问题) 记录始终映射到同一向量 id，替换时可定位删除"""
    q_hash = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
    return f"log_{base_id}_{q_hash}"


def merge_log_records(existing_df: pd.DataFrame, new_records: list[dict]):
    """按 图片+问题 查重合并日志记录。

    决策表：
      - 键不存在        → added（追加）
      - 存在且新分数更高 → replaced（整体替换，时间戳用新的）
      - 存在且分数相等   → 保留时间戳更晚的一条（同分保新）
      - 存在且新分数更低 → dropped（保留旧值）

    Returns:
        (merged_df, {"added": int, "replaced": int, "dropped": int}, replaced_keys)
        replaced_keys: 实际发生替换的 (id, question) 键列表，供向量库定位删除旧向量
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

    for rec in new_records:
        key = record_key(rec["id"], rec["question"])
        new_score = float(rec.get("correctness_score", 0.0))
        new_ts = str(rec.get("timestamp", ""))

        if key not in key_to_idx:
            stats["added"] += 1
            row = pd.DataFrame([{c: rec.get(c, "") for c in LOG_COLUMNS}], columns=LOG_COLUMNS)
            df = pd.concat([df, row], ignore_index=True)
            key_to_idx[key] = len(df) - 1
            continue

        idx = key_to_idx[key]
        old_score = float(df.at[idx, "correctness_score"] or 0.0)
        old_ts = str(df.at[idx, "timestamp"] or "")

        if new_score > old_score:
            stats["replaced"] += 1
            replaced_keys.append((rec["id"], rec["question"]))
            for c in LOG_COLUMNS:
                df.at[idx, c] = rec.get(c, "")
        elif new_score == old_score and new_ts > old_ts:
            stats["replaced"] += 1
            replaced_keys.append((rec["id"], rec["question"]))
            for c in LOG_COLUMNS:
                df.at[idx, c] = rec.get(c, "")
        else:
            stats["dropped"] += 1

    return df, stats, replaced_keys
```

- [ ] **Step 5: 运行确认通过**

Run: `python -m pytest tests/test_log_merge.py -v`
Expected: PASS（7 个测试全过）

- [ ] **Step 6: 提交**

```bash
git add tests/__init__.py tests/conftest.py tests/test_log_merge.py agent/rag/membership/log_manager.py
git commit -m "feat(membership): 测试基础设施 + 图片+问题查重/高分替换纯逻辑与确定性向量id"
```

---

### Task 2: 综合向量文本构建（含切片内容与回答、截断）

**Files:**
- Modify: `agent/rag/membership/log_manager.py`
- Test: `tests/test_log_text.py`（新建）

**Interfaces:**
- Produces:
  - `build_log_doc_text(question: str, slices_content: list[str], answer: str, slice_char_limit: int = 200, total_char_limit: int = 1500) -> str`
    - 每片截断至 `slice_char_limit` 字符，整体截断至 `total_char_limit` 字符
    - 格式：`Question: {q}\nRetrieved Slices: {片间\n分隔}\nAnswer: {a}`
  - `fetch_slices_content(slice_store, slices_id_list: list[str]) -> str`
    - 从切片库 `get_by_ids` 取内容，`\n` 拼接；空 id 列表返回 `""`；缺失切片跳过
    - 放在 log_manager.py 中（与向量文本构建同属日志记录构建职责），供初始化/入库脚本共用

- [ ] **Step 1: 写失败测试**

```python
# tests/test_log_text.py
from rag.membership.log_manager import build_log_doc_text, fetch_slices_content
from tests.conftest import FakeSliceStore


def test_three_sections():
    t = build_log_doc_text("q?", ["slice one"], "answer text")
    assert t == "Question: q?\nRetrieved Slices: slice one\nAnswer: answer text"


def test_multiple_slices_join_with_newline():
    t = build_log_doc_text("q?", ["s1", "s2"], "a")
    assert "Retrieved Slices: s1\ns2\nAnswer: a" in t


def test_per_slice_truncation():
    t = build_log_doc_text("q?", ["x" * 300], "a", slice_char_limit=200)
    # 每片截 200 字符
    assert "x" * 200 in t
    assert "x" * 201 not in t


def test_total_truncation():
    t = build_log_doc_text("q?" + "y" * 300, ["s"], "a", total_char_limit=500)
    assert len(t) <= 500


def test_empty_answer():
    t = build_log_doc_text("q?", ["s"], "")
    assert t.endswith("Answer: ")


def test_fetch_slices_content_joins():
    store = FakeSliceStore()
    assert fetch_slices_content(store, ["s1", "s2"]) == "content_of_s1\ncontent_of_s2"


def test_fetch_slices_content_empty():
    store = FakeSliceStore()
    assert fetch_slices_content(store, []) == ""
    assert fetch_slices_content(store, ["", " "]) == ""
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_log_text.py -v`
Expected: FAIL（ImportError）

- [ ] **Step 3: 实现**

在 `agent/rag/membership/log_manager.py` 的 `merge_log_records` 之后追加：

```python
def build_log_doc_text(
    question: str,
    slices_content: list[str],
    answer: str,
    slice_char_limit: int = 200,
    total_char_limit: int = 1500,
) -> str:
    """构建隶属度相似度的综合向量文本：问题 + 切片内容 + 回答。

    切片内容每片截断至 slice_char_limit 字符；整体截断至 total_char_limit 字符，
    防止超出嵌入模型的输入上限。
    """
    truncated_slices = [s[:slice_char_limit] for s in slices_content if s]
    slices_part = "\n".join(truncated_slices)
    doc = f"Question: {question}\nRetrieved Slices: {slices_part}\nAnswer: {answer}"
    return doc[:total_char_limit]


def fetch_slices_content(slice_store, slices_id_list: list[str]) -> str:
    """从切片库按 id 批量取切片内容，换行拼接（缺失切片跳过）"""
    valid = [s for s in slices_id_list if s and str(s).strip()]
    if not valid:
        return ""
    result = slice_store.get_by_ids(valid)
    documents = result.get("documents", []) or []
    return "\n".join(d for d in documents if d)
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_log_text.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add tests/test_log_text.py agent/rag/membership/log_manager.py
git commit -m "feat(membership): 综合向量文本构建（问题+切片内容+回答，含截断）与切片内容获取"
```

---

### Task 3: LogManager 新 schema 适配（CSV 读写/补列/向量文本/确定性 id）

**Files:**
- Modify: `agent/rag/membership/log_manager.py`
  - `__init__`（空 DataFrame 列 → `LOG_COLUMNS`，4 处）
  - `_init_force_reload` / `_init_clear_vector_db` / `_init_incremental`（补列列表 → `LOG_COLUMNS`）
  - `add_single_record`（向量文本用 `build_log_doc_text`，id 用 `generate_log_vector_id`，metadata 增加 `correct`、`retrieved_slices_content`）
  - `_build_log_texts`（同上）
  - `_generate_unique_id` → 删除（由 `generate_log_vector_id` 替代）

**Interfaces:**
- Consumes: Task 1 的 `LOG_COLUMNS`/`generate_log_vector_id`；Task 2 的 `build_log_doc_text`
- Produces: `LogManager` 现有公开接口签名不变（`log_df`/`logs_collection`/`add_single_record`/`reload_to_vector_db`/`_incrementally_add_new_records`），内部用新 schema 与向量文本

- [ ] **Step 1: 写失败测试（mock Chroma，验证写入的文本与 id）**

```python
# tests/test_log_manager_schema.py
import pandas as pd
import pytest

from rag.membership.log_manager import LOG_COLUMNS, build_log_doc_text, generate_log_vector_id
from tests.conftest import NoopCollection


def make_manager(tmp_path, fake_col, monkeypatch):
    """构造 LogManager，注入 fake 向量集合（不落盘真实 Chroma）"""
    from rag.membership import log_manager as lm
    monkeypatch.setattr(lm.rag_config, "persist_directory", str(tmp_path / "chroma"))
    monkeypatch.setattr(lm.rag_config, "data_path", str(tmp_path))
    monkeypatch.setattr(lm.rag_config, "log_path", str(tmp_path / "logs.csv"))
    monkeypatch.setattr(lm.rag_config, "md5_log_store_path", str(tmp_path / "md5.txt"))
    m = lm.LogManager()
    m._logs_collection = fake_col
    m._chroma_mgr._collection = fake_col
    return m


def test_new_empty_df_has_new_columns(tmp_path, monkeypatch):
    m = make_manager(tmp_path, NoopCollection(), monkeypatch)
    assert list(m.log_df.columns) == LOG_COLUMNS


def test_add_single_record_uses_combined_text_and_deterministic_id(tmp_path, monkeypatch):
    fake = NoopCollection()
    m = make_manager(tmp_path, fake, monkeypatch)
    entry = {
        "id": "img1.png", "question": "q?", "retrieved_slices": "s1|s2",
        "retrieved_slices_content": "slice one\nslice two", "predicted_text": "ans",
        "correct": 1, "correctness_score": 0.9, "timestamp": "2026-08-07 10:00:00",
    }
    m.add_single_record(entry)
    call = fake.calls[0]
    assert call["ids"][0] == generate_log_vector_id("img1.png", "q?")
    assert call["texts"][0] == build_log_doc_text("q?", ["slice one", "slice two"], "ans")
    assert call["metadatas"][0]["correct"] == "1"
    assert call["metadatas"][0]["correctness_score"] == "0.9"


def test_build_log_texts_skips_empty_question(tmp_path, monkeypatch):
    fake = NoopCollection()
    m = make_manager(tmp_path, fake, monkeypatch)
    df = pd.DataFrame([
        {"id": "a.png", "question": "q1", "retrieved_slices": "s1", "retrieved_slices_content": "c1",
         "predicted_text": "p1", "correct": 1, "correctness_score": 0.5, "timestamp": "t1"},
        {"id": "b.png", "question": "", "retrieved_slices": "", "retrieved_slices_content": "",
         "predicted_text": "", "correct": 0, "correctness_score": 0.0, "timestamp": ""},
    ], columns=LOG_COLUMNS)
    texts, metadatas, ids = m._build_log_texts(df)
    assert len(texts) == 1
    assert ids[0] == generate_log_vector_id("a.png", "q1")
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_log_manager_schema.py -v`
Expected: FAIL（新列缺失 / id 不符）

- [ ] **Step 3: 实现**

在 `log_manager.py` 中替换以下片段：

3a. `__init__`、`_init_force_reload`、`_init_clear_vector_db`、`_init_incremental` 中 4 处空/补列 DataFrame 均改为：

```python
        self._log_df = pd.DataFrame(columns=LOG_COLUMNS)
```

（补列循环处改为：）

```python
            for col in LOG_COLUMNS:
                if col not in self._log_df.columns:
                    self._log_df[col] = ""
```

3b. `_init_force_reload` 中 error_df 不变（error 日志仍用旧 schema）。

3c. `add_single_record` 改为：

```python
    def add_single_record(self, log_entry: dict):
        """添加单条记录到日志和向量库（综合向量：问题+切片内容+回答）"""
        slices_content = str(log_entry.get("retrieved_slices_content", "")).splitlines()
        doc_text = build_log_doc_text(
            log_entry["question"], slices_content, log_entry.get("predicted_text", "")
        )
        unique_id = generate_log_vector_id(str(log_entry["id"]), log_entry["question"])

        self._logs_collection.add_texts(
            texts=[doc_text],
            metadatas=[{
                "id": str(log_entry["id"]),
                "question": log_entry["question"],
                "retrieved_slices": log_entry.get("retrieved_slices", ""),
                "retrieved_slices_content": log_entry.get("retrieved_slices_content", ""),
                "correctness_score": str(log_entry.get("correctness_score", 0.0)),
                "correct": str(log_entry.get("correct", 0)),
                "predicted_text": log_entry.get("predicted_text", ""),
                "timestamp": log_entry.get("timestamp", ""),
            }],
            ids=[unique_id],
        )
```

3d. `_build_log_texts` 的 doc_text 与 metadata 部分改为：

```python
            slices_content = (
                str(row["retrieved_slices_content"]).splitlines()
                if not pd.isna(row["retrieved_slices_content"]) else []
            )
            doc_text = build_log_doc_text(
                str(row["question"]), slices_content, str(row["predicted_text"])
            )
            texts.append(doc_text)
            metadatas.append({
                "id": str(row["id"]),
                "question": str(row["question"]),
                "retrieved_slices": (
                    str(row["retrieved_slices"]) if not pd.isna(row["retrieved_slices"]) else ""
                ),
                "retrieved_slices_content": (
                    str(row["retrieved_slices_content"])
                    if not pd.isna(row["retrieved_slices_content"]) else ""
                ),
                "correctness_score": (
                    str(row["correctness_score"]) if not pd.isna(row["correctness_score"]) else "0.0"
                ),
                "correct": str(row["correct"]) if not pd.isna(row["correct"]) else "0",
                "predicted_text": (
                    str(row["predicted_text"]) if not pd.isna(row["predicted_text"]) else ""
                ),
                "timestamp": (
                    str(row["timestamp"]) if not pd.isna(row["timestamp"]) else ""
                ),
            })
            ids.append(generate_log_vector_id(str(row["id"]), str(row["question"])))
```

3e. 删除 `_generate_unique_id` 静态方法（含其唯一调用处替换，见 3c/3d）。

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_log_manager_schema.py -v`
Expected: PASS

- [ ] **Step 5: 全量回归**

Run: `python -m pytest tests/ -v`
Expected: 全过（Task 1-3 测试合计）

- [ ] **Step 6: 提交**

```bash
git add tests/test_log_manager_schema.py agent/rag/membership/log_manager.py
git commit -m "feat(membership): LogManager 新schema（correct/切片内容列、综合向量、确定性id）"
```

---

### Task 4: 向量库批量 upsert（新增 add + 替换 delete/add）

**Files:**
- Modify: `agent/rag/membership/log_manager.py`
- Test: `tests/test_log_manager_upsert.py`（新建）

**Interfaces:**
- Consumes: Task 1 `merge_log_records` / `generate_log_vector_id`；Task 2 `build_log_doc_text`
- Produces:
  - `LogManager.upsert_records_to_vector_db(records: list[dict], replaced_keys: list[tuple[str, str]]) -> dict`
    - 对 `records`（新加/替换后的全部记录）批量 add（确定性 id）
    - 对 `replaced_keys`（被替换记录的 图片+问题 键）按旧 id `delete(ids=[...])` 再 add 新向量，避免重复
    - 返回 `{"added": int, "deleted": int}`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_log_manager_upsert.py
import pandas as pd
import pytest

from rag.membership.log_manager import (
    LOG_COLUMNS, generate_log_vector_id, merge_log_records,
)
from tests.conftest import NoopCollection
from tests.test_log_manager_schema import make_manager


def _rec(id_, q, score, **kw):
    r = {"id": id_, "question": q, "retrieved_slices": "s1", "retrieved_slices_content": "c",
         "predicted_text": "p", "correct": 1, "correctness_score": score, "timestamp": "t"}
    r.update(kw)
    return r


def test_upsert_adds_new_and_replaces_old(tmp_path, monkeypatch):
    fake = NoopCollection()
    m = make_manager(tmp_path, fake, monkeypatch)
    # 库中已有 (img1, q1) 分数 0.5
    df = pd.DataFrame([_rec("img1.png", "q1", 0.5)], columns=LOG_COLUMNS)
    merged, stats, replaced_keys = merge_log_records(
        df, [_rec("img1.png", "q1", 0.9, timestamp="new")]
    )
    assert replaced_keys == [("img1.png", "q1")]

    m.upsert_records_to_vector_db(
        merged.to_dict("records"),
        replaced_keys=replaced_keys,
    )

    old_id = generate_log_vector_id("img1.png", "q1")
    assert fake.deleted == [old_id]
    # 所有 merged 记录都被 add（含替换后的新向量）
    assert len(fake.calls) == 1
    assert fake.calls[0]["ids"] == [old_id]


def test_upsert_empty_records_noop(tmp_path, monkeypatch):
    fake = NoopCollection()
    m = make_manager(tmp_path, fake, monkeypatch)
    result = m.upsert_records_to_vector_db([], [])
    assert result == {"added": 0, "deleted": 0}
    assert fake.calls == []
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_log_manager_upsert.py -v`
Expected: FAIL（AttributeError: no attribute 'upsert_records_to_vector_db'）

- [ ] **Step 3: 实现**

在 `log_manager.py` 中 `add_single_record` 之后追加：

```python
    def upsert_records_to_vector_db(
        self,
        records: list[dict],
        replaced_keys: list[tuple[str, str]] = None,
    ) -> dict:
        """批量 upsert 记录到向量库。

        Args:
            records: 合并后需入库的全部记录（含新增与替换后的新内容）
            replaced_keys: 被替换记录的 (图片, 问题) 键列表——先删旧向量再写新向量

        Returns:
            {"added": int, "deleted": int}
        """
        replaced_keys = replaced_keys or []
        deleted = 0
        if replaced_keys:
            old_ids = [
                generate_log_vector_id(id_, q) for id_, q in replaced_keys
            ]
            try:
                self._logs_collection.delete(ids=old_ids)
                deleted = len(old_ids)
            except Exception as e:
                logger.error("[upsert] 删除旧向量失败: %s", e)

        if not records:
            return {"added": 0, "deleted": deleted}

        texts, metadatas, ids = self._build_log_texts(pd.DataFrame(records, columns=LOG_COLUMNS))
        added = self._chroma_mgr.add_texts_in_batches(
            texts=texts, metadatas=metadatas, ids=ids, desc="批量 upsert 日志"
        )
        return {"added": added, "deleted": deleted}
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_log_manager_upsert.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add tests/test_log_manager_upsert.py agent/rag/membership/log_manager.py
git commit -m "feat(membership): 向量库批量upsert（新增add + 替换delete/add）"
```

---

### Task 5: 暂存区写入（staging.py）

**Files:**
- Create: `agent/rag/membership/staging.py`
- Modify: `agent/rag/core/config.py`（新增 `pending_log_path` 属性）
- Test: `tests/test_staging.py`（新建）

**Interfaces:**
- Produces:
  - `PENDING_COLUMNS = ["id", "question", "retrieved_slices", "predicted_text", "correct", "correctness_score", "timestamp"]`
  - `append_pending_record(record: dict, pending_path: str = None) -> None`
    - 线程安全（`utils.thread_lock.lock`）；文件不存在则先写表头；追加一行；`pending_path=None` 时用 `rag_config.pending_log_path`
  - `count_pending_records(pending_path: str = None) -> int`（文件缺失返回 0）
  - `clear_pending_records(pending_path: str = None, backup: bool = True) -> str|None`
    - backup=True 时先复制为 `pending_records_backup_<时间戳>.csv`，再清空文件（仅保留表头）；返回备份路径

- [ ] **Step 1: 写失败测试**

```python
# tests/test_staging.py
import os
import pandas as pd
import pytest

from rag.membership.staging import (
    PENDING_COLUMNS, append_pending_record, count_pending_records, clear_pending_records,
)


def test_append_creates_file_with_header(tmp_path):
    p = str(tmp_path / "pending.csv")
    append_pending_record(
        {"id": "a.png", "question": "q1", "retrieved_slices": "s1", "predicted_text": "p",
         "correct": 1, "correctness_score": 0.8, "timestamp": "2026-08-07 10:00:00"},
        pending_path=p,
    )
    df = pd.read_csv(p)
    assert list(df.columns) == PENDING_COLUMNS
    assert len(df) == 1


def test_append_multiple_rows(tmp_path):
    p = str(tmp_path / "pending.csv")
    for i in range(3):
        append_pending_record(
            {"id": f"a{i}.png", "question": f"q{i}", "retrieved_slices": "s", "predicted_text": "p",
             "correct": 1, "correctness_score": 0.5, "timestamp": "t"},
            pending_path=p,
        )
    assert count_pending_records(p) == 3


def test_count_missing_file_is_zero(tmp_path):
    assert count_pending_records(str(tmp_path / "none.csv")) == 0


def test_clear_backs_up_and_truncates(tmp_path):
    p = str(tmp_path / "pending.csv")
    for i in range(2):
        append_pending_record(
            {"id": f"a{i}.png", "question": f"q{i}", "retrieved_slices": "s", "predicted_text": "p",
             "correct": 1, "correctness_score": 0.5, "timestamp": "t"},
            pending_path=p,
        )
    backup = clear_pending_records(p)
    assert backup is not None and os.path.exists(backup)
    assert pd.read_csv(backup).shape[0] == 2      # 备份有 2 条
    assert count_pending_records(p) == 0          # 原文件只剩表头
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_staging.py -v`
Expected: FAIL（ModuleNotFoundError: rag.membership.staging）

- [ ] **Step 3: 实现 staging.py**

```python
"""
运行期新记录暂存区管理
在线系统每会话产生一条记录 → append_pending_record() 追加到 pending_records.csv
批量入库脚本（agent/data/flush_pending_logs.py）检测条数 → 超阈值 → 合并入库 → 清空
"""
import os
import shutil
from datetime import datetime

import pandas as pd

from rag.core.config import rag_config
from utils.thread_lock import lock

PENDING_COLUMNS = [
    "id", "question", "retrieved_slices", "predicted_text",
    "correct", "correctness_score", "timestamp",
]


def _resolve_path(pending_path: str = None) -> str:
    return pending_path or rag_config.pending_log_path


def append_pending_record(record: dict, pending_path: str = None) -> None:
    """线程安全地追加一条暂存记录（在线系统每次会话调用）"""
    path = _resolve_path(pending_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = {c: record.get(c, "") for c in PENDING_COLUMNS}
    with lock:
        if not os.path.exists(path):
            pd.DataFrame(columns=PENDING_COLUMNS).to_csv(path, index=False, encoding="utf-8")
        pd.DataFrame([row]).to_csv(path, index=False, mode="a", header=False, encoding="utf-8")


def count_pending_records(pending_path: str = None) -> int:
    """检测暂存区待入库记录条数（文件缺失返回 0）"""
    path = _resolve_path(pending_path)
    if not os.path.exists(path):
        return 0
    try:
        df = pd.read_csv(path)
        return len(df)
    except Exception:
        return 0


def clear_pending_records(pending_path: str = None, backup: bool = True):
    """清空暂存区（仅保留表头）；backup=True 时先备份为 *_backup_<时间戳>.csv

    Returns:
        str | None: 备份文件路径（未备份或清空失败时 None）
    """
    path = _resolve_path(pending_path)
    if not os.path.exists(path):
        return None
    backup_path = None
    if backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.replace(".csv", f"_backup_{ts}.csv")
        shutil.copy(path, backup_path)
    with lock:
        pd.DataFrame(columns=PENDING_COLUMNS).to_csv(path, index=False, encoding="utf-8")
    return backup_path
```

- [ ] **Step 4: config.py 增加 pending 路径**

`agent/rag/core/config.py` 的 `RagConfig` 类中，`log_path` 属性之后追加：

```python
    @property
    def pending_log_path(self) -> str:
        return get_abs_path(f"{self.data_path}/pending_records.csv")
```

- [ ] **Step 5: 运行确认通过**

Run: `python -m pytest tests/test_staging.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add tests/test_staging.py agent/rag/membership/staging.py agent/rag/core/config.py
git commit -m "feat(membership): 暂存区写入（线程安全追加/计数/备份清空）"
```

---

### Task 6: 初始化脚本（init_membership_logs.py）

**Files:**
- Create: `agent/data/init_membership_logs.py`
- Test: `tests/test_init_script.py`（新建，用 fake evaluator/judge，不调真实 API）

**Interfaces:**
- Consumes: Task 1-4 的 `LogManager`（`merge_log_records`/`build_log_doc_text`/`upsert_records_to_vector_db`/`reload_to_vector_db`）、`rag_config`
- Produces:
  - `run_init(csv_path: str, max_rows: int, evaluator, judge, slice_store, log_manager, force: bool) -> dict`
    - 返回 `{"total": int, "correct_count": int, "final_count": int, "elapsed_sec": float}`
    - evaluator 需要 `call_vlm_agent(image, question, ground_truth) -> (score, text, slice_ids)`
    - judge 需要 `judge_and_filter(test_df, logs) -> (correct_logs, error_logs)`
    - slice_store 需要 `get_by_ids(ids) -> {"ids": [...], "documents": [...], "metadatas": [...]}`（documents 即切片内容）
  - CLI：`python agent/data/init_membership_logs.py --csv <path> --max-rows N --force`
    - 默认 `--csv agent/dataset_split/test.csv`、`--max-rows 20000`
    - 不带 `--force` 时交互确认（`input()` 询问）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_init_script.py
import pandas as pd
import pytest

from rag.membership.log_manager import LOG_COLUMNS
from tests.conftest import NoopCollection, FakeSliceStore, load_script


class FakeEvaluator:
    """返回固定分数/回答/切片id"""
    def __init__(self, score_map):
        self.score_map = score_map

    def call_vlm_agent(self, image_path, question, ground_truth=None):
        row = self.score_map[(image_path, question)]
        return row["score"], row["text"], row["slices"]


class FakeJudge:
    def __init__(self, correct_ids):
        self.correct_ids = set(correct_ids)

    def judge_and_filter(self, test_df, logs):
        correct, error = [], []
        for log in logs:
            (correct if log["id"] in self.correct_ids else error).append(log)
        return correct, error


def _fake_dataframe():
    rows = [
        {"id": "a.png", "image": "a.png", "question": "q1", "answer": "ans1"},
        {"id": "b.png", "image": "b.png", "question": "q2", "answer": "ans2"},
        {"id": "c.png", "image": "c.png", "question": "q3", "answer": "ans3"},
    ]
    return pd.DataFrame(rows)


def test_run_init_keeps_only_correct(tmp_path, monkeypatch):
    from rag.membership import log_manager as lm
    init_mod = load_script("init_membership_logs")

    fake = _fake_dataframe()
    csv_path = tmp_path / "test.csv"
    fake.to_csv(csv_path, index=False)
    score_map = {("a.png", "q1"): {"score": 0.9, "text": "pa", "slices": ["s1", "s2"]},
                 ("b.png", "q2"): {"score": 0.8, "text": "pb", "slices": ["s3"]},
                 ("c.png", "q3"): {"score": 0.7, "text": "pc", "slices": []}}
    evaluator = FakeEvaluator(score_map)
    judge = FakeJudge(correct_ids={"a.png", "c.png"})   # b 错误

    monkeypatch.setattr(lm.rag_config, "persist_directory", str(tmp_path / "chroma"))
    monkeypatch.setattr(lm.rag_config, "data_path", str(tmp_path))
    monkeypatch.setattr(lm.rag_config, "log_path", str(tmp_path / "logs.csv"))
    monkeypatch.setattr(lm.rag_config, "md5_log_store_path", str(tmp_path / "md5.txt"))
    manager = lm.LogManager()
    manager._logs_collection = NoopCollection()
    manager._chroma_mgr._collection = manager._logs_collection

    result = init_mod.run_init(
        csv_path=str(csv_path), max_rows=100, evaluator=evaluator,
        judge=judge, slice_store=FakeSliceStore(), log_manager=manager, force=True,
    )
    assert result["total"] == 3
    assert result["correct_count"] == 2
    assert result["final_count"] == 2
    df = pd.read_csv(tmp_path / "logs.csv")
    assert len(df) == 2
    assert set(df["id"]) == {"a.png", "c.png"}
    assert df.iloc[0]["correct"] == 1
    assert "content_of_s1" in df.iloc[0]["retrieved_slices_content"]
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_init_script.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现 init_membership_logs.py**

```python
"""
隶属度历史日志初始化脚本（手动运行）
从数据集重跑预测 → LLM 判断 → 只保留预测正确的记录 → 综合向量全量入库

用法:
    python agent/data/init_membership_logs.py                    # 默认 test.csv, 20000 条
    python agent/data/init_membership_logs.py --csv agent/dataset_split/test.csv --max-rows 5000 --force
"""
import argparse
import os
import shutil
import sys
import time
from datetime import datetime

# 确保 agent/ 在 sys.path（脚本在 agent/data/ 下运行时）
_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

import pandas as pd

from rag.core.config import rag_config
from rag.membership.log_manager import (
    LOG_COLUMNS, LogManager, merge_log_records, fetch_slices_content,
)
from rag.membership.vlm_evaluator import VlmEvaluator
from rag.membership.llm_judge import LlmJudge
from rag.stores.slice_store import SliceStore
from utils.logger_handler import logger


def run_init(csv_path, max_rows, evaluator, judge, slice_store, log_manager, force=False):
    """执行初始化。evaluator/judge/slice_store 可注入 fake 用于测试。"""
    if not force:
        answer = input(f"将重建隶属度日志库（备份后清空重建），确认继续? [y/N]: ")
        if answer.strip().lower() != "y":
            print("已取消")
            return None

    start = time.time()
    df = pd.read_csv(csv_path, nrows=max_rows)
    logger.info("读取数据集 %s，共 %d 条", csv_path, len(df))

    # 1. 备份旧日志
    if os.path.exists(rag_config.log_path):
        backup = rag_config.log_path.replace(".csv", f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        shutil.copy(rag_config.log_path, backup)
        logger.info("旧日志已备份到 %s", backup)

    # 2. 逐条预测
    logs = []
    for _, row in df.iterrows():
        try:
            score, text, slice_ids = evaluator.call_vlm_agent(
                row["image"], row["question"], row.get("answer")
            )
            logs.append({
                "id": str(row["id"]),
                "question": str(row["question"]),
                "retrieved_slices": "|".join(slice_ids or []),
                "predicted_text": text,
                "correctness_score": score,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
        except Exception as e:
            logger.error("预测失败 [%s] %s: %s", row.get("id"), row.get("question"), e)

    # 3. LLM 判断正确性，只保留正确的
    correct_logs, error_logs = judge.judge_and_filter(df, logs)
    logger.info("LLM 判断: 正确 %d / 错误 %d", len(correct_logs), len(error_logs))

    # 4. 补切片内容 + correct 列 → 查重合并
    for log in correct_logs:
        log["correct"] = 1
        slice_ids = str(log.get("retrieved_slices", "")).split("|")
        log["retrieved_slices_content"] = fetch_slices_content(slice_store, slice_ids)
    merged_df, stats, _ = merge_log_records(
        pd.DataFrame(columns=LOG_COLUMNS), correct_logs
    )

    # 5. 写 CSV
    os.makedirs(os.path.dirname(rag_config.log_path), exist_ok=True)
    merged_df.to_csv(rag_config.log_path, index=False, encoding="utf-8")
    log_manager.log_df = merged_df

    # 6. 重建向量库
    log_manager.reload_to_vector_db()

    elapsed = time.time() - start
    logger.info("初始化完成: 预测 %d | 正确 %d | 入库 %d | 耗时 %.1f 秒",
                len(logs), len(correct_logs), len(merged_df), elapsed)
    print(f"✅ 初始化完成: 预测 {len(logs)} | 正确 {len(correct_logs)} | "
          f"入库 {len(merged_df)} | 耗时 {elapsed:.1f} 秒")
    return {"total": len(logs), "correct_count": len(correct_logs),
            "final_count": len(merged_df), "elapsed_sec": elapsed}


def main():
    parser = argparse.ArgumentParser(description="隶属度日志初始化：只保留预测正确的会话记录")
    parser.add_argument("--csv", default="agent/dataset_split/test.csv")
    parser.add_argument("--max-rows", type=int, default=20000)
    parser.add_argument("--force", action="store_true", help="跳过确认，直接备份并重建")
    args = parser.parse_args()

    run_init(
        csv_path=args.csv,
        max_rows=args.max_rows,
        evaluator=VlmEvaluator(),
        judge=LlmJudge(),
        slice_store=SliceStore(),
        log_manager=LogManager(force_full_reload=True),
        force=args.force,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_init_script.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add tests/test_init_script.py agent/data/init_membership_logs.py
git commit -m "feat(membership): 初始化脚本（test.csv重跑+LLM判断+仅正确记录入库）"
```

---

### Task 7: 批量入库脚本（flush_pending_logs.py）

**Files:**
- Create: `agent/data/flush_pending_logs.py`
- Test: `tests/test_flush_script.py`（新建）

**Interfaces:**
- Consumes: Task 1 `merge_log_records`；Task 2 `fetch_slices_content`；Task 3 `LogManager`；Task 5 `staging`（count/clear）
- Produces:
  - `run_flush(pending_path: str, log_manager, slice_store, threshold: int) -> dict`
    - 返回 `{"pending": int, "added": int, "replaced": int, "dropped": int, "flushed": bool}`
    - pending 条数 < threshold → 打印跳过，`flushed=False`（不清空）
    - ≥ threshold → 读 pending → 补切片内容 → 查重合并 → CSV 保存 + 向量库 upsert → 备份并清空 pending
  - CLI：`python agent/data/flush_pending_logs.py --threshold 50`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_flush_script.py
import pandas as pd
import pytest

from rag.membership.log_manager import LOG_COLUMNS
from rag.membership.staging import PENDING_COLUMNS, append_pending_record
from tests.conftest import NoopCollection, FakeSliceStore, load_script


def _manager(tmp_path, monkeypatch):
    from rag.membership import log_manager as lm
    monkeypatch.setattr(lm.rag_config, "persist_directory", str(tmp_path / "chroma"))
    monkeypatch.setattr(lm.rag_config, "data_path", str(tmp_path))
    monkeypatch.setattr(lm.rag_config, "log_path", str(tmp_path / "logs.csv"))
    monkeypatch.setattr(lm.rag_config, "md5_log_store_path", str(tmp_path / "md5.txt"))
    m = lm.LogManager()
    m._logs_collection = NoopCollection()
    m._chroma_mgr._collection = m._logs_collection
    return m


def _pending_rec(id_, q, score):
    return {"id": id_, "question": q, "retrieved_slices": "s1", "predicted_text": "p",
            "correct": 1, "correctness_score": score, "timestamp": "2026-08-07 12:00:00"}


def test_flush_below_threshold_skips(tmp_path, monkeypatch):
    flush = load_script("flush_pending_logs")
    p = str(tmp_path / "pending.csv")
    append_pending_record(_pending_rec("a.png", "q1", 0.8), pending_path=p)
    result = flush.run_flush(p, _manager(tmp_path, monkeypatch), FakeSliceStore(), threshold=50)
    assert result["flushed"] is False
    assert result["pending"] == 1
    assert pd.read_csv(p).shape[0] == 1   # 未清空


def test_flush_merges_replaces_and_clears(tmp_path, monkeypatch):
    flush = load_script("flush_pending_logs")
    m = _manager(tmp_path, monkeypatch)
    # 库中已有 (a.png, q1) 分数 0.5
    m.log_df = pd.DataFrame([{
        "id": "a.png", "question": "q1", "retrieved_slices": "s0", "retrieved_slices_content": "old",
        "predicted_text": "old", "correct": 1, "correctness_score": 0.5, "timestamp": "2026-08-07 08:00:00",
    }], columns=LOG_COLUMNS)

    p = str(tmp_path / "pending.csv")
    append_pending_record(_pending_rec("a.png", "q1", 0.9), pending_path=p)      # 替换
    append_pending_record(_pending_rec("b.png", "q2", 0.7), pending_path=p)      # 新增
    append_pending_record(_pending_rec("a.png", "q1", 0.4), pending_path=p)      # 丢弃

    result = flush.run_flush(p, m, FakeSliceStore(), threshold=2)
    assert result["flushed"] is True
    assert result["added"] == 1
    assert result["replaced"] == 1
    assert result["dropped"] == 1

    df = pd.read_csv(tmp_path / "logs.csv")
    assert len(df) == 2
    row_a = df[df["id"] == "a.png"].iloc[0]
    assert row_a["correctness_score"] == 0.9
    assert "content_of_s1" in row_a["retrieved_slices_content"]
    assert pd.read_csv(p).shape[0] == 0   # 已清空（仅表头）
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_flush_script.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现 flush_pending_logs.py**

```python
"""
隶属度历史日志批量入库脚本（手动运行）
检测暂存区（agent/data/pending_records.csv）待入库记录条数，
超过阈值后按 图片+问题 查重合并（新加/高分替换）并批量写入日志库与向量库，
成功后备份并清空暂存区。

用法:
    python agent/data/flush_pending_logs.py                 # 默认阈值 50
    python agent/data/flush_pending_logs.py --threshold 20
"""
import argparse
import os
import sys

_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

import pandas as pd

from rag.core.config import rag_config
from rag.membership.log_manager import (
    LOG_COLUMNS, LogManager, merge_log_records, fetch_slices_content,
)
from rag.membership.staging import count_pending_records, clear_pending_records
from rag.stores.slice_store import SliceStore
from utils.logger_handler import logger


def run_flush(pending_path, log_manager, slice_store, threshold=50):
    """执行批量入库。返回统计 dict；pending 条数 < threshold 时不入库、不清空。"""
    pending_count = count_pending_records(pending_path)
    if pending_count < threshold:
        print(f"⏭ 暂存区记录 {pending_count} 条 < 阈值 {threshold}，跳过入库")
        return {"pending": pending_count, "added": 0, "replaced": 0,
                "dropped": 0, "flushed": False}

    df = pd.read_csv(pending_path)
    records = df.to_dict("records")

    # 补切片内容
    for rec in records:
        slice_ids = str(rec.get("retrieved_slices", "")).split("|")
        rec["retrieved_slices_content"] = fetch_slices_content(slice_store, slice_ids)
        rec["correct"] = int(float(rec.get("correct", 0))) if not pd.isna(rec.get("correct")) else 0
        rec["correctness_score"] = float(rec.get("correctness_score", 0.0))

    # 查重合并（与现有日志库）；replaced_keys 为实际被替换的 (id, question) 键
    merged_df, stats, replaced_keys = merge_log_records(log_manager.log_df, records)

    # 写 CSV
    os.makedirs(os.path.dirname(rag_config.log_path), exist_ok=True)
    merged_df.to_csv(rag_config.log_path, index=False, encoding="utf-8")
    log_manager.log_df = merged_df

    # 向量库 upsert（先删被替换旧向量，再批量 add 全量新记录）
    log_manager.upsert_records_to_vector_db(merged_df.to_dict("records"), replaced_keys)

    # 备份并清空暂存区
    backup = clear_pending_records(pending_path)
    logger.info("暂存区已清空，备份: %s", backup)

    print(f"✅ 入库完成: 新增 {stats['added']} | 替换 {stats['replaced']} | "
          f"丢弃 {stats['dropped']} | 暂存 {pending_count} 条已清空")
    return {"pending": pending_count, **stats, "flushed": True}


def main():
    parser = argparse.ArgumentParser(description="隶属度日志批量入库")
    parser.add_argument("--threshold", type=int, default=50, help="待入库条数阈值")
    args = parser.parse_args()
    run_flush(rag_config.pending_log_path, LogManager(), SliceStore(),
              threshold=args.threshold)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_flush_script.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add tests/test_flush_script.py agent/data/flush_pending_logs.py
git commit -m "feat(membership): 批量入库脚本（阈值检测+查重合并+备份清空）"
```

---

### Task 8: 召回策略对比实验（compare_recall_strategies.py）

**Files:**
- Create: `agent/data/compare_recall_strategies.py`
- Test: `tests/test_compare_script.py`（新建）

**Interfaces:**
- Consumes: `rag_config`；日志集合（`similarity_search_with_relevance_scores`）；`SliceStore.get_by_ids`
- Produces:
  - `strategy_direct(query: str, log_collection, k: int) -> tuple[list[str], float]` — 上下文（日志 metadata 的切片内容+回答），返回 (context_lines, elapsed_sec)
  - `strategy_via_slices(query: str, log_collection, slice_store, k: int) -> tuple[list[str], float]` — 从日志拿切片 id → `SliceStore.get_by_ids` 取内容，返回 (context_lines, elapsed_sec)
  - `run_compare(queries: list[str], log_collection, slice_store, k: int = None) -> dict`
    - 返回 `{"direct": {"total": .., "avg": .., "p50": .., "p95": ..}, "via_slices": {...}, "winner": "direct"|"via_slices"}`
  - CLI：`python agent/data/compare_recall_strategies.py --n 5000 --k 50`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_compare_script.py
import pytest

from tests.conftest import load_script


class FakeLogCollection:
    """top k 日志；metadata 含 retrieved_slices_content(内容) 与 retrieved_slices(id 列表)"""
    def __init__(self, latency=0.01):
        self.latency = latency

    def similarity_search_with_relevance_scores(self, query, k=None):
        import time
        time.sleep(self.latency)
        from types import SimpleNamespace
        n = k or 3
        docs = []
        for i in range(n):
            docs.append((SimpleNamespace(metadata={
                "retrieved_slices_content": f"content_{i}",
                "predicted_text": f"answer_{i}",
                "retrieved_slices": f"slice_{i}",
            }), 0.9 - i * 0.01))
        return docs


class FakeSliceStore:
    def __init__(self, latency=0.02):
        self.latency = latency

    def get_by_ids(self, ids):
        import time
        time.sleep(self.latency)
        return {"ids": ids, "documents": [f"slice_content_{i}" for i in ids],
                "metadatas": [{} for _ in ids]}


def test_strategies_produce_context_and_timing():
    mod = load_script("compare_recall_strategies")
    coll = FakeLogCollection()
    direct_ctx, direct_t = mod.strategy_direct("q", coll, k=3)
    assert len(direct_ctx) == 3
    assert "content_0" in direct_ctx[0]
    assert direct_t > 0

    via_ctx, via_t = mod.strategy_via_slices("q", coll, FakeSliceStore(), k=3)
    assert len(via_ctx) == 3
    assert "slice_content_0" in via_ctx[0]
    assert via_t > direct_t   # fake 切片库更慢


def test_run_compare_returns_winner():
    mod = load_script("compare_recall_strategies")
    result = mod.run_compare(["q1", "q2"], FakeLogCollection(latency=0.005),
                             FakeSliceStore(latency=0.03), k=3)
    assert set(result.keys()) == {"direct", "via_slices", "winner"}
    assert result["winner"] == "direct"
    assert result["direct"]["total"] > 0
    assert result["direct"]["avg"] < result["via_slices"]["avg"]
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_compare_script.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现 compare_recall_strategies.py**

```python
"""
召回方式对比实验（手动运行）
对比两种从历史日志库召回上下文的方式：
  策略1 直接日志库召回：query → 日志库向量检索 → 上下文 = 日志 metadata 切片内容+回答（不查切片库）
  策略2 id→切片库召回：query → 日志库向量检索 → 切片 id → 切片库 get_by_ids 取内容
批量跑 N 次查询，输出耗时对比，选效率更高的策略。

用法:
    python agent/data/compare_recall_strategies.py --n 5000 --k 50
"""
import argparse
import os
import sys
import time

_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

import pandas as pd

from rag.core.config import rag_config
from rag.membership.log_manager import LogManager
from rag.stores.slice_store import SliceStore


def strategy_direct(query: str, log_collection, k: int):
    """策略1：只查日志库。返回 (上下文行列表, 耗时秒)"""
    start = time.perf_counter()
    results = log_collection.similarity_search_with_relevance_scores(query, k=k)
    context = []
    for doc, _ in results:
        meta = doc.metadata
        context.append(meta.get("retrieved_slices_content", ""))
        context.append(meta.get("predicted_text", ""))
    return context, time.perf_counter() - start


def strategy_via_slices(query: str, log_collection, slice_store, k: int):
    """策略2：日志库拿切片 id → 切片库取内容。返回 (上下文行列表, 耗时秒)"""
    start = time.perf_counter()
    results = log_collection.similarity_search_with_relevance_scores(query, k=k)
    slice_ids = []
    for doc, _ in results:
        meta = doc.metadata
        slice_ids.extend(str(meta.get("retrieved_slices", "")).split("|"))
    slice_ids = [s for s in slice_ids if s]
    context = []
    if slice_ids:
        got = slice_store.get_by_ids(slice_ids)
        context = [d for d in (got.get("documents") or []) if d]
    return context, time.perf_counter() - start


def _stats(times: list[float]) -> dict:
    import numpy as np
    a = np.array(times)
    return {
        "total": float(a.sum()),
        "avg": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
    }


def run_compare(queries: list[str], log_collection, slice_store, k: int = None):
    """跑两策略各 N 次，返回耗时统计与胜者"""
    k = k or rag_config.membership_k
    direct_times, via_times = [], []
    for q in queries:
        _, dt = strategy_direct(q, log_collection, k)
        _, vt = strategy_via_slices(q, log_collection, slice_store, k)
        direct_times.append(dt)
        via_times.append(vt)

    direct_stats = _stats(direct_times)
    via_stats = _stats(via_times)
    winner = "direct" if direct_stats["avg"] <= via_stats["avg"] else "via_slices"
    return {"direct": direct_stats, "via_slices": via_stats, "winner": winner}


def main():
    parser = argparse.ArgumentParser(description="召回方式对比实验")
    parser.add_argument("--n", type=int, default=5000, help="查询次数")
    parser.add_argument("--k", type=int, default=None, help="日志检索 top k（默认 rag_config.membership_k）")
    parser.add_argument("--csv", default="agent/dataset_split/test.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, nrows=args.n)
    queries = df["question"].tolist()

    log_manager = LogManager()
    slice_store = SliceStore()
    result = run_compare(queries, log_manager.logs_collection, slice_store, k=args.k)

    print(f"对比实验完成（{len(queries)} 次查询, top k={args.k or rag_config.membership_k}）")
    print(f"策略1 直接日志库召回: 总 {result['direct']['total']:.2f}s | "
          f"平均 {result['direct']['avg']*1000:.2f}ms | P50 {result['direct']['p50']*1000:.2f}ms | "
          f"P95 {result['direct']['p95']*1000:.2f}ms")
    print(f"策略2 id→切片库召回: 总 {result['via_slices']['total']:.2f}s | "
          f"平均 {result['via_slices']['avg']*1000:.2f}ms | P50 {result['via_slices']['p50']*1000:.2f}ms | "
          f"P95 {result['via_slices']['p95']*1000:.2f}ms")
    winner = result["winner"]
    print(f"🏆 推荐策略: {'策略1 直接日志库召回' if winner == 'direct' else '策略2 id→切片库召回'}"
          f"（平均耗时更低）")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_compare_script.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add tests/test_compare_script.py agent/data/compare_recall_strategies.py
git commit -m "feat(membership): 召回方式对比实验（直接召回 vs id→切片库召回）"
```

---

### Task 9: 全量回归与收尾

**Files:**
- Modify: 无（仅验证）

- [ ] **Step 1: 全量测试**

Run: `python -m pytest tests/ -v`
Expected: 全部 PASS（Task 1-8 的测试文件合计）

- [ ] **Step 2: 三个脚本 CLI 冒烟（--help）**

Run:
```bash
python agent/data/init_membership_logs.py --help
python agent/data/flush_pending_logs.py --help
python agent/data/compare_recall_strategies.py --help
```
Expected: 三脚本打印 usage 无报错

- [ ] **Step 3: 手动流程演练（可选，真实初始化）**

Run: `python agent/data/init_membership_logs.py --csv agent/dataset_split/test.csv --max-rows 100 --force`
Expected: 打印「初始化完成: 预测 100 | 正确 N | 入库 N | 耗时 X 秒」；`agent/data/rag_feedback_logs.csv` 重建为新 schema（8 列）；旧文件有 `_backup_<时间戳>` 备份

- [ ] **Step 4: 提交（若有遗留）**

```bash
git add -A
git commit -m "chore(membership): 隶属度日志改造收尾验证"
```

---

## 自审记录（写入时执行）

- **Spec 覆盖**：初始化（Task 6）✓；图片+问题查重/高分替换（Task 1）✓；综合向量含回答（Task 2/3）✓；手动入库+阈值（Task 7 + Task 5 暂存）✓；召回对比实验（Task 8）✓；0/1 与连续分都存（Task 1 schema + Task 3 metadata）✓
- **类型一致性**：`merge_log_records`（返回 `(merged_df, stats, replaced_keys)`）在 Task 1/4/6/7 的解包处一致；`generate_log_vector_id`/`build_log_doc_text`/`fetch_slices_content`/`upsert_records_to_vector_db`/`append_pending_record`/`count_pending_records`/`clear_pending_records`/`run_init`/`run_flush`/`run_compare` 签名在跨任务引用处一致
- **导入路径**：脚本测试统一用 conftest 的 `load_script`（importlib 加载），`fetch_slices_content` 放 `rag.membership.log_manager`，避免 `agent.data` 命名空间包导入的脆弱性
- **测试隔离**：所有测试注入 fake（NoopCollection/FakeEvaluator/FakeJudge/FakeSliceStore）与临时目录，不触碰真实 Chroma/API/数据文件
