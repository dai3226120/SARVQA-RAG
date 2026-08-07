# tests/test_cache_system_schema.py
"""evaluate_and_log 在线路径写新 schema 的集成测试：
LLM 判断通过后必须补 correct=1 与 retrieved_slices_content（综合向量在线路径）"""
import pandas as pd

from rag.membership import log_manager as lm
from rag.membership import cache_system as cs
from tests.conftest import NoopCollection, FakeSliceStore


def make_system(tmp_path, monkeypatch):
    """构造 SemanticCacheSystem，注入 fake 向量集合 / VLM / LLM / 切片库"""
    monkeypatch.setattr(lm.rag_config, "persist_directory", str(tmp_path / "chroma"))
    monkeypatch.setattr(lm.rag_config, "data_path", str(tmp_path))
    # log_path 是只读 property（data_path + log_name 派生），改底层的 log_name
    monkeypatch.setattr(lm.rag_config, "log_name", "logs.csv")
    monkeypatch.setattr(lm.rag_config, "md5_log_store_path", str(tmp_path / "md5.txt"))
    monkeypatch.setattr(lm, "Chroma", lambda **kwargs: NoopCollection())
    monkeypatch.setattr(cs, "SliceStore", lambda *a, **k: FakeSliceStore())

    class FakeVlm:
        def call_vlm_agent(self, image, question, answer=None):
            return (0.9, "predicted answer", ["s1", "s2"])

    class FakeJudge:
        def judge_and_filter(self, test_df, new_logs):
            return new_logs, []  # 全部判为正确

    monkeypatch.setattr(cs, "VlmEvaluator", lambda: FakeVlm())
    monkeypatch.setattr(cs, "LlmJudge", lambda: FakeJudge())

    sys = cs.SemanticCacheSystem()
    fake = NoopCollection()
    sys._log_manager._logs_collection = fake
    sys._log_manager._chroma_mgr._collection = fake
    return sys


def test_evaluate_and_log_writes_new_schema_columns(tmp_path, monkeypatch):
    sys = make_system(tmp_path, monkeypatch)
    test_df = pd.DataFrame([{
        "id": "img1.png", "image": "img1.png", "question": "q1", "answer": "ans",
    }])

    sys.evaluate_and_log(test_df, max_workers=2)

    # CSV 已按新 schema 落盘：correct=1、切片内容非空
    df = pd.read_csv(tmp_path / "logs.csv")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["question"] == "q1"
    assert row["correct"] == 1
    assert "content_of_s1" in row["retrieved_slices_content"]

    # 向量 metadata 同步：correct="1"、切片内容完整
    call = sys._log_manager._logs_collection.calls[0]
    assert call["metadatas"][0]["correct"] == "1"
    assert call["metadatas"][0]["retrieved_slices_content"] == "content_of_s1\ncontent_of_s2"
    # 综合向量文本含回答（在线路径接入 build_log_doc_text）
    assert "Answer: predicted answer" in call["texts"][0]
