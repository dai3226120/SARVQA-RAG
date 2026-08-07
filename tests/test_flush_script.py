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


def test_flush_keeps_pending_when_vector_db_upsert_fails(tmp_path, monkeypatch):
    """设计条款：向量库失败 → 保留 pending 不清空、flushed=False（CSV 已更新属可接受）"""
    flush = load_script("flush_pending_logs")
    m = _manager(tmp_path, monkeypatch)
    # 伪造 upsert 失败：added=0 与合并后条数不符
    monkeypatch.setattr(m, "upsert_records_to_vector_db",
                        lambda *a, **k: {"added": 0, "deleted": 0})

    p = str(tmp_path / "pending.csv")
    append_pending_record(_pending_rec("a.png", "q1", 0.8), pending_path=p)

    result = flush.run_flush(p, m, FakeSliceStore(), threshold=1)

    assert result["flushed"] is False
    assert result["pending"] == 1
    assert pd.read_csv(p).shape[0] == 1       # pending 未被清空（保留供下次重试）
    assert len(pd.read_csv(tmp_path / "logs.csv")) == 1   # CSV 合并成功已写入


def test_script_self_injects_agent_and_project_root(monkeypatch):
    """回归防护：脚本必须自行注入 agent/ 与项目根（utils/model 包位于项目根）。

    注意不能只断言 load_script 成功——conftest 已注入 agent/ 与项目根，
    且 rag/utils 模块已被本测试模块导入（sys.modules 缓存），单注入时
    load 也可能成功。故先移除 sys.path 中的 agent/ 与项目根再加载，
    断言脚本把两者都插回 sys.path。
    """
    import sys
    from pathlib import Path

    root = str(Path(__file__).resolve().parents[1])
    agent = str(Path(root) / "agent")
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p not in (root, agent)])
    assert root not in sys.path and agent not in sys.path  # 前置条件成立

    mod = load_script("flush_pending_logs")

    assert agent in sys.path, "脚本未自行注入 agent/"
    assert root in sys.path, "脚本未自行注入项目根（utils/model 包会 ImportError）"
    assert hasattr(mod, "run_flush")


def test_cli_loadable_standalone_subprocess():
    """CLI 独立运行不得因 sys.path 缺失崩溃（python agent/data/flush_pending_logs.py --help）。

    子进程为全新解释器，无 conftest 注入、无 sys.modules 缓存，能真实暴露单注入缺陷。
    注意本机 Windows 存在既有 ortools WinDLL 冲突（conftest.py 顶部注释记载，
    init_membership_logs.py 同样受影响），--help 可能以 WinError 127 退出；
    但只要 stderr 不含 "No module named"，即证明 utils/model 等根路径包已可导入。
    """
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "agent" / "data" / "flush_pending_logs.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True, text=True, timeout=300,
    )
    assert "ModuleNotFoundError" not in proc.stderr, f"脚本 import 崩溃:\n{proc.stderr}"
    assert "No module named" not in proc.stderr, f"sys.path 注入缺失:\n{proc.stderr}"
