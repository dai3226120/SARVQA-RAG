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
    # log_path 是只读 property（data_path + log_name 派生），改底层的 log_name
    monkeypatch.setattr(lm.rag_config, "log_name", "logs.csv")
    monkeypatch.setattr(lm.rag_config, "md5_log_store_path", str(tmp_path / "md5.txt"))
    # 避免 LogManager.__init__ 创建真实 Chroma（sqlite 在 Windows 上持锁）
    monkeypatch.setattr(lm, "Chroma", lambda **kwargs: NoopCollection())
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
