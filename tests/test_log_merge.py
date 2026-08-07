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
