"""日志库语义去重 + 容量上限测试（同图去重规格见 docs/superpowers/specs/2026-08-09-log-dedup-cap-design.md）"""
import pandas as pd
import pytest

from rag.core.config import rag_config
from rag.membership.log_manager import (
    LOG_COLUMNS, merge_log_records,
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
