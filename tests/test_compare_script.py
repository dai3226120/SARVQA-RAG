import pytest

from tests.conftest import load_script


class FakeLogCollection:
    """top k 日志；metadata 含 retrieved_slices_content(内容) 与 retrieved_slices(id 列表)"""
    def __init__(self, latency=0.01):
        self.latency = latency

    def similarity_search_with_relevance_scores(self, query, k=None):
        import time
        time.sleep(self.latency)
        from types import SimpleNamespace
        n = k or 3
        docs = []
        for i in range(n):
            docs.append((SimpleNamespace(metadata={
                "retrieved_slices_content": f"content_{i}",
                "predicted_text": f"answer_{i}",
                "retrieved_slices": f"slice_{i}",
            }), 0.9 - i * 0.01))
        return docs


class FakeSliceStore:
    def __init__(self, latency=0.02):
        self.latency = latency

    def get_by_ids(self, ids):
        import time
        time.sleep(self.latency)
        return {"ids": ids, "documents": [f"slice_content_{i}" for i in ids],
                "metadatas": [{} for _ in ids]}


def test_strategies_produce_context_and_timing():
    mod = load_script("compare_recall_strategies")
    coll = FakeLogCollection()
    direct_ctx, direct_t = mod.strategy_direct("q", coll, k=3)
    # 策略1 每条日志追加 2 行（切片内容+回答），3 条日志 → 6 行
    assert len(direct_ctx) == 6
    assert "content_0" in direct_ctx[0]
    assert direct_t > 0

    via_ctx, via_t = mod.strategy_via_slices("q", coll, FakeSliceStore(), k=3)
    assert len(via_ctx) == 3
    # fake 的切片 id 形如 "slice_0"，get_by_ids 产出 f"slice_content_{id}" → "slice_content_slice_0"
    assert "slice_content_slice_0" in via_ctx[0]
    assert via_t > direct_t   # fake 切片库更慢


def test_run_compare_returns_winner():
    mod = load_script("compare_recall_strategies")
    result = mod.run_compare(["q1", "q2"], FakeLogCollection(latency=0.005),
                             FakeSliceStore(latency=0.03), k=3)
    assert set(result.keys()) == {"direct", "via_slices", "winner"}
    assert result["winner"] == "direct"
    assert result["direct"]["total"] > 0
    assert result["direct"]["avg"] < result["via_slices"]["avg"]
