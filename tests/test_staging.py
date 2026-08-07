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
