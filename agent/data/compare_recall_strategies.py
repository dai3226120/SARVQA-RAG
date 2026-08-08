"""
召回方式对比实验（手动运行）
对比两种从历史日志库召回上下文的方式（均使用全量精确检索，非 HNSW 近似）：
  策略1 直接日志库召回：query → 日志库全量精确 top-k → 上下文 = 日志 metadata 切片内容+回答（不查切片库）
  策略2 id→切片库召回：query → 日志库全量精确 top-k → 切片 id → 切片库 get_by_ids 取内容
批量跑 N 次查询，输出耗时对比，选效率更高的策略。

实现要点：
  - 全量精确：启动时导出日志库全部向量到 numpy 矩阵，每次查询矩阵乘 + argsort 得精确 top-k，
    两策略共享同一份 top-k 结果，对比的是取数路径（日志 metadata vs 切片库）差异
  - 批量预嵌入：所有 query 预先批量嵌入（GPU batch 比逐条快约 5 倍），嵌入不计入策略耗时
  - 计时拆分为: 公共检索(精确 top-k + 日志 metadata 取数) / 策略2 额外切片库取数

用法:
    python agent/data/compare_recall_strategies.py --n 5000 --k 50
"""
# Windows 下 pyarrow(pandas 间接依赖) 先加载会导致 ortools DLL 加载失败(WinError 127)。
# 必须先于 pandas 导入 k_means_constrained，让 ortools 扩展 DLL 先进入进程。
import k_means_constrained  # noqa: F401

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

import numpy as np
import pandas as pd

from rag.core.config import rag_config
from rag.membership.log_manager import LogManager
from rag.stores.slice_store import SliceStore


def export_all_vectors(raw_collection) -> tuple[np.ndarray, np.ndarray]:
    """分页导出日志库全部向量，返回 (归一化向量矩阵 (N,d), id 数组)"""
    total = raw_collection.count()
    vecs, ids = [], []
    offset = 0
    while offset < total:
        r = raw_collection.get(offset=offset, limit=500, include=["embeddings"])
        vecs.extend(r["embeddings"])
        ids.extend(r["ids"])
        offset += 500
    V = np.array(vecs, dtype=np.float32)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)  # 预归一化，余弦=点积
    return V, np.array(ids)


def exact_top_k(V: np.ndarray, ids: np.ndarray, qe: np.ndarray, k: int) -> list[str]:
    """全量精确余弦 top-k，返回日志 id 列表（按相似度降序）"""
    scores = V @ qe
    return ids[np.argsort(scores)[-k:][::-1]].tolist()


def strategy_direct(raw_collection, top_k_ids: list[str]):
    """策略1：从日志库按 id 精确取 metadata 组装上下文（不做向量检索）。返回 (上下文行列表, 耗时秒)"""
    start = time.perf_counter()
    metas = raw_collection.get(ids=top_k_ids, include=["metadatas"])["metadatas"]
    context = []
    for meta in metas:
        context.append(meta.get("retrieved_slices_content", ""))
        context.append(meta.get("predicted_text", ""))
    return context, time.perf_counter() - start


def strategy_via_slices(raw_collection, slice_store, top_k_ids: list[str]):
    """策略2：日志 metadata 拿切片 id → 切片库按 id 取内容。返回 (上下文行列表, 耗时秒)"""
    start = time.perf_counter()
    metas = raw_collection.get(ids=top_k_ids, include=["metadatas"])["metadatas"]
    slice_ids = []
    for meta in metas:
        slice_ids.extend(str(meta.get("retrieved_slices", "")).split("|"))
    # 去重并保持顺序：不同日志可共享同一切片，Chroma get 要求 id 唯一
    slice_ids = list(dict.fromkeys(s for s in slice_ids if s))
    context = []
    if slice_ids:
        got = slice_store.get_by_ids(slice_ids)
        context = [d for d in (got.get("documents") or []) if d]
    return context, time.perf_counter() - start


def _stats(times: list[float]) -> dict:
    a = np.array(times)
    return {
        "total": float(a.sum()),
        "avg": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
    }


def run_compare(
    queries: list[str],
    log_manager: LogManager,
    slice_store: SliceStore,
    k: int = None,
) -> dict:
    """批量预嵌入 + 全量精确 top-k，两策略共享同一份检索结果，对比取数路径耗时"""
    k = k or rag_config.membership_k
    raw_col = log_manager.logs_collection._collection
    embeddings = log_manager._embeddings

    # 1. 导出全量向量（精确检索基础）
    print(f"导出日志库全量向量（{raw_col.count()} 条）...", flush=True)
    V, ids = export_all_vectors(raw_col)

    # 2. 批量预嵌入（GPU batch，远快于逐条）
    print(f"批量嵌入 {len(queries)} 条 query ...", flush=True)
    embed_start = time.perf_counter()
    qe_list = embeddings.embed_documents(queries)
    embed_elapsed = time.perf_counter() - embed_start

    # 3. 循环：共享精确 top-k，分别计时两策略取数
    direct_times, via_times, retrieve_times = [], [], []
    for q, qe in zip(queries, qe_list):
        qe = np.array(qe, dtype=np.float32)
        qe = qe / np.linalg.norm(qe)

        t0 = time.perf_counter()
        top_k_ids = exact_top_k(V, ids, qe, k)
        retrieve_times.append((time.perf_counter() - t0) * 1000)

        _, dt = strategy_direct(raw_col, top_k_ids)
        _, vt = strategy_via_slices(raw_col, slice_store, top_k_ids)
        direct_times.append(dt * 1000)
        via_times.append(vt * 1000)

    return {
        "embed": {"total": embed_elapsed, "avg_per_query": embed_elapsed / len(queries)},
        "retrieve": _stats(retrieve_times),
        "direct": _stats(direct_times),
        "via_slices": _stats(via_times),
        "winner": "direct" if _stats(direct_times)["avg"] <= _stats(via_times)["avg"] else "via_slices",
        "n_logs": len(ids),
    }


def main():
    parser = argparse.ArgumentParser(description="召回方式对比实验（全量精确检索）")
    parser.add_argument("--n", type=int, default=5000, help="查询次数")
    parser.add_argument("--k", type=int, default=None, help="日志检索 top k（默认 rag_config.membership_k）")
    parser.add_argument("--csv", default="agent/dataset_split/test.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, nrows=args.n)
    queries = df["question"].tolist()

    log_manager = LogManager()
    slice_store = SliceStore()
    result = run_compare(queries, log_manager, slice_store, k=args.k)

    k = args.k or rag_config.membership_k
    print(f"\n对比实验完成（{len(queries)} 次查询, top k={k}, 日志库 {result['n_logs']} 条, 全量精确检索）")
    print(f"批量嵌入: 总 {result['embed']['total']:.2f}s | 单条等效 {result['embed']['avg_per_query']*1000:.2f}ms")
    print(f"公共检索(精确top-k): 平均 {result['retrieve']['avg']:.2f}ms | P50 {result['retrieve']['p50']:.2f}ms | "
          f"P95 {result['retrieve']['p95']:.2f}ms")
    print(f"策略1 取数(日志metadata): 平均 {result['direct']['avg']:.2f}ms | P50 {result['direct']['p50']:.2f}ms | "
          f"P95 {result['direct']['p95']:.2f}ms")
    print(f"策略2 取数(切片库get_by_ids): 平均 {result['via_slices']['avg']:.2f}ms | "
          f"P50 {result['via_slices']['p50']:.2f}ms | P95 {result['via_slices']['p95']:.2f}ms")
    winner = result["winner"]
    print(f"🏆 推荐策略: {'策略1 直接日志库召回' if winner == 'direct' else '策略2 id→切片库召回'}"
          f"（取数平均耗时更低）")


if __name__ == "__main__":
    main()
