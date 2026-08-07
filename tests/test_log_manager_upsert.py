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
