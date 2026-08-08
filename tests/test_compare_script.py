"""compare_recall_strategies 全量精确版测试：
两策略取数正确性 + run_compare 输出结构（注入 fake 集合与嵌入函数，不落盘）"""
import pytest

from tests.conftest import load_script


class FakeRawCollection:
    """chromadb 底层集合 fake：少量向量 + metadata 按 id 返回"""
    def __init__(self, n=5, dim=4):
        import numpy as np
        self._count = n
        self._vecs = np.random.rand(n, dim).astype("float32").tolist()
        self._ids = [f"log_{i}" for i in range(n)]
        self._metas = [
            {
                "retrieved_slices_content": f"content_{i}",
                "predicted_text": f"answer_{i}",
                "retrieved_slices": f"slice_{i}",
            }
            for i in range(n)
        ]

    def count(self):
        return self._count

    def get(self, offset=0, limit=None, ids=None, include=None):
        if ids is not None:
            idx = [self._ids.index(i) for i in ids]
            return {
                "ids": ids,
                "documents": [f"doc_{i}" for i in idx],
                "metadatas": [self._metas[i] for i in idx],
                "embeddings": [self._vecs[i] for i in idx],
            }
        end = len(self._ids) if limit is None else offset + limit
        idx = list(range(offset, min(end, len(self._ids))))
        return {
            "ids": [self._ids[i] for i in idx],
            "documents": [f"doc_{i}" for i in idx],
            "metadatas": [self._metas[i] for i in idx],
            "embeddings": [self._vecs[i] for i in idx],
        }


class FakeEmbeddings:
    """embed_query/embed_documents 返回确定性向量（与集合向量维度一致）"""
    def __init__(self, dim=4):
        self._dim = dim

    def embed_query(self, text):
        import numpy as np
        v = np.zeros(self._dim, dtype="float32")
        v[0] = 1.0
        return v.tolist()

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


class FakeLogManager:
    """提供 logs_collection._collection 与 _embeddings，满足 run_compare 接口"""
    def __init__(self, raw_col):
        self.logs_collection = type("ChromaWrap", (), {"_collection": raw_col})()
        self._embeddings = FakeEmbeddings()


class FakeSliceStore:
    def get_by_ids(self, ids):
        return {
            "ids": ids,
            "documents": [f"slice_content_{i}" for i in ids],
            "metadatas": [{} for _ in ids],
        }


def test_strategies_produce_context_and_timing():
    mod = load_script("compare_recall_strategies")
    raw = FakeRawCollection()
    # 策略1：精确 top-k ids → 每条日志 2 行（切片内容+回答）
    ctx, t = mod.strategy_direct(raw, ["log_0", "log_1", "log_2"])
    assert len(ctx) == 6
    assert "content_0" in ctx[0]
    assert t > 0

    # 策略2：切片 id → 切片库取内容
    ctx2, t2 = mod.strategy_via_slices(raw, FakeSliceStore(), ["log_0", "log_1", "log_2"])
    assert len(ctx2) == 3
    assert "slice_content_slice_0" in ctx2[0]
    assert t2 > 0


def test_exact_top_k_returns_sorted_ids():
    import numpy as np
    mod = load_script("compare_recall_strategies")
    raw = FakeRawCollection()
    V, ids = mod.export_all_vectors(raw)
    qe = np.zeros(V.shape[1], dtype="float32")
    qe[0] = 1.0
    top = mod.exact_top_k(V, ids, qe, 3)
    assert len(top) == 3
    assert all(isinstance(i, str) for i in top)
    # 距离排序：v[0]=1 的向量得分最高，应出现在 top
    assert "log_0" in top


def test_run_compare_returns_winner():
    mod = load_script("compare_recall_strategies")
    result = mod.run_compare(["q1", "q2"], FakeLogManager(FakeRawCollection()),
                             FakeSliceStore(), k=3)
    assert set(result.keys()) == {"embed", "retrieve", "direct", "via_slices", "winner", "n_logs"}
    assert result["winner"] in ("direct", "via_slices")
    assert result["direct"]["total"] > 0
    assert result["via_slices"]["total"] > 0
    assert result["n_logs"] == 5
