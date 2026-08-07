# tests/test_log_manager_dedup.py
"""加载期去重（_deduplicate_log_df）测试：查重键 = 图片+问题 联合键，同键保留最新"""
import pandas as pd

from rag.membership.log_manager import LOG_COLUMNS
from tests.conftest import NoopCollection


def make_manager(tmp_path, monkeypatch):
    """构造 LogManager，注入 fake 向量集合（不落盘真实 Chroma）"""
    from rag.membership import log_manager as lm
    monkeypatch.setattr(lm.rag_config, "persist_directory", str(tmp_path / "chroma"))
    monkeypatch.setattr(lm.rag_config, "data_path", str(tmp_path))
    # log_path 是只读 property（data_path + log_name 派生），改底层的 log_name
    monkeypatch.setattr(lm.rag_config, "log_name", "logs.csv")
    monkeypatch.setattr(lm.rag_config, "md5_log_store_path", str(tmp_path / "md5.txt"))
    monkeypatch.setattr(lm, "Chroma", lambda **kwargs: NoopCollection())
    m = lm.LogManager()
    m._logs_collection = NoopCollection()
    m._chroma_mgr._collection = m._logs_collection
    return m


def _row(id_, q, ts, score=0.5):
    return {"id": id_, "question": q, "retrieved_slices": "s1",
            "retrieved_slices_content": "c", "predicted_text": "p", "correct": 1,
            "correctness_score": score, "timestamp": ts}


def test_same_id_different_question_both_kept(tmp_path, monkeypatch):
    """同图片不同问题：两条记录都保留（联合查重语义）"""
    m = make_manager(tmp_path, monkeypatch)
    m.log_df = pd.DataFrame([
        _row("img1.png", "q1", "2026-08-07 10:00:00"),
        _row("img1.png", "q2", "2026-08-07 11:00:00"),
    ], columns=LOG_COLUMNS)

    m._deduplicate_log_df()

    assert len(m.log_df) == 2
    assert set(m.log_df["question"]) == {"q1", "q2"}
    # 去重结果已写回 CSV
    assert pd.read_csv(tmp_path / "logs.csv").shape[0] == 2


def test_same_id_same_question_keeps_latest(tmp_path, monkeypatch):
    """同图片同问题：只保留时间戳最新的一条"""
    m = make_manager(tmp_path, monkeypatch)
    m.log_df = pd.DataFrame([
        _row("img1.png", "q1", "2026-08-07 09:00:00", score=0.5),
        _row("img1.png", "q1", "2026-08-07 12:00:00", score=0.9),
    ], columns=LOG_COLUMNS)

    m._deduplicate_log_df()

    assert len(m.log_df) == 1
    assert m.log_df.iloc[0]["correctness_score"] == 0.9  # 保留新记录
    df = pd.read_csv(tmp_path / "logs.csv")
    assert df.shape[0] == 1
    assert df.iloc[0]["timestamp"] == "2026-08-07 12:00:00"
