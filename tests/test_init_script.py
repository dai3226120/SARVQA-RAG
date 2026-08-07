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
    # log_path 是只读 property（data_path + log_name 派生），改底层的 log_name
    monkeypatch.setattr(lm.rag_config, "log_name", "logs.csv")
    monkeypatch.setattr(lm.rag_config, "md5_log_store_path", str(tmp_path / "md5.txt"))
    # LogManager.__init__ 会创建真实 Chroma，其 sqlite 在 Windows 上持锁，
    # 导致 reload 时 clear_and_rebuild 的 shutil.rmtree 失败（WinError 32）。
    # 阻止 __init__ 实例化真实 Chroma；重建路径仍在 chroma_manager 内部走真实 Chroma（tmp 内）。
    monkeypatch.setattr(lm, "Chroma", lambda **kwargs: NoopCollection())
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


def test_main_constructs_default_log_manager(monkeypatch):
    """回归防护：main() 必须用默认模式 LogManager()。

    若用 LogManager(force_full_reload=True)，参数求值先于 run_init 执行，
    _init_force_reload 会先截断旧日志，导致 run_init 的「备份旧日志」拿到截断文件。
    """
    import sys

    init_mod = load_script("init_membership_logs")

    class FakeLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    captured = {}

    def fake_run_init(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(init_mod, "LogManager", FakeLM)
    monkeypatch.setattr(init_mod, "VlmEvaluator", lambda: object())
    monkeypatch.setattr(init_mod, "LlmJudge", lambda: object())
    monkeypatch.setattr(init_mod, "SliceStore", lambda: object())
    monkeypatch.setattr(init_mod, "run_init", fake_run_init)
    monkeypatch.setattr(sys, "argv", ["init_membership_logs.py", "--force"])

    init_mod.main()

    mgr = captured["log_manager"]
    assert isinstance(mgr, FakeLM)
    assert mgr.kwargs.get("force_full_reload") in (None, False)
    assert captured["force"] is True
