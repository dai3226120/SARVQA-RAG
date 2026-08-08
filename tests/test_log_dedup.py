"""日志库语义去重 + 容量上限测试（同图去重规格见 docs/superpowers/specs/2026-08-09-log-dedup-cap-design.md）"""
import pandas as pd

from rag.core.config import rag_config
from rag.membership.log_manager import (
    LOG_COLUMNS, merge_log_records, dedup_in_batch, find_semantic_duplicates,
    enforce_log_cap,
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
