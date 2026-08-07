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
    # log_path 是只读 property（data_path + log_name 派生），改底层的 log_name
    monkeypatch.setattr(lm.rag_config, "log_name", "logs.csv")
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
