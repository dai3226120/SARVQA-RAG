"""
召回方式对比实验（手动运行）
对比两种从历史日志库召回上下文的方式：
  策略1 直接日志库召回：query → 日志库向量检索 → 上下文 = 日志 metadata 切片内容+回答（不查切片库）
  策略2 id→切片库召回：query → 日志库向量检索 → 切片 id → 切片库 get_by_ids 取内容
批量跑 N 次查询，输出耗时对比，选效率更高的策略。

用法:
    python agent/data/compare_recall_strategies.py --n 5000 --k 50
"""
import argparse
import os
import sys
import time

# 确保 agent/ 与项目根在 sys.path（脚本在 agent/data/ 下运行时；utils 包位于项目根）
_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_AGENT_DIR)
for _p in (_PROJECT_ROOT, _AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

from rag.core.config import rag_config
from rag.membership.log_manager import LogManager
from rag.stores.slice_store import SliceStore


def strategy_direct(query: str, log_collection, k: int):
    """策略1：只查日志库。返回 (上下文行列表, 耗时秒)"""
    start = time.perf_counter()
    results = log_collection.similarity_search_with_relevance_scores(query, k=k)
    context = []
    for doc, _ in results:
        meta = doc.metadata
        context.append(meta.get("retrieved_slices_content", ""))
        context.append(meta.get("predicted_text", ""))
    return context, time.perf_counter() - start


def strategy_via_slices(query: str, log_collection, slice_store, k: int):
    """策略2：日志库拿切片 id → 切片库取内容。返回 (上下文行列表, 耗时秒)"""
    start = time.perf_counter()
    results = log_collection.similarity_search_with_relevance_scores(query, k=k)
    slice_ids = []
    for doc, _ in results:
        meta = doc.metadata
        slice_ids.extend(str(meta.get("retrieved_slices", "")).split("|"))
    slice_ids = [s for s in slice_ids if s]
    context = []
    if slice_ids:
        got = slice_store.get_by_ids(slice_ids)
        context = [d for d in (got.get("documents") or []) if d]
    return context, time.perf_counter() - start


def _stats(times: list[float]) -> dict:
    import numpy as np
    a = np.array(times)
    return {
        "total": float(a.sum()),
        "avg": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
    }


def run_compare(queries: list[str], log_collection, slice_store, k: int = None):
    """跑两策略各 N 次，返回耗时统计与胜者"""
    k = k or rag_config.membership_k
    direct_times, via_times = [], []
    for q in queries:
        _, dt = strategy_direct(q, log_collection, k)
        _, vt = strategy_via_slices(q, log_collection, slice_store, k)
        direct_times.append(dt)
        via_times.append(vt)

    direct_stats = _stats(direct_times)
    via_stats = _stats(via_times)
    winner = "direct" if direct_stats["avg"] <= via_stats["avg"] else "via_slices"
    return {"direct": direct_stats, "via_slices": via_stats, "winner": winner}


def main():
    parser = argparse.ArgumentParser(description="召回方式对比实验")
    parser.add_argument("--n", type=int, default=5000, help="查询次数")
    parser.add_argument("--k", type=int, default=None, help="日志检索 top k（默认 rag_config.membership_k）")
    parser.add_argument("--csv", default="agent/dataset_split/test.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, nrows=args.n)
    queries = df["question"].tolist()

    log_manager = LogManager()
    slice_store = SliceStore()
    result = run_compare(queries, log_manager.logs_collection, slice_store, k=args.k)

    print(f"对比实验完成（{len(queries)} 次查询, top k={args.k or rag_config.membership_k}）")
    print(f"策略1 直接日志库召回: 总 {result['direct']['total']:.2f}s | "
          f"平均 {result['direct']['avg']*1000:.2f}ms | P50 {result['direct']['p50']*1000:.2f}ms | "
          f"P95 {result['direct']['p95']*1000:.2f}ms")
    print(f"策略2 id→切片库召回: 总 {result['via_slices']['total']:.2f}s | "
          f"平均 {result['via_slices']['avg']*1000:.2f}ms | P50 {result['via_slices']['p50']*1000:.2f}ms | "
          f"P95 {result['via_slices']['p95']*1000:.2f}ms")
    winner = result["winner"]
    print(f"🏆 推荐策略: {'策略1 直接日志库召回' if winner == 'direct' else '策略2 id→切片库召回'}"
          f"（平均耗时更低）")


if __name__ == "__main__":
    main()
