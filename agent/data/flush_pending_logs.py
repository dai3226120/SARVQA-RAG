"""
隶属度历史日志批量入库脚本（手动运行）
检测暂存区（agent/data/pending_records.csv）待入库记录条数，
超过阈值后按 图片+问题 查重合并（新加/高分替换）并批量写入日志库与向量库，
成功后备份并清空暂存区。

语义去重（同图问题级相似度 ≥ dedup.sim_threshold 视为问法变体）：
  1. 批内合并：待入库记录之间同图高相似 → 高分优先合并
  2. 库内查重：对每条新记录召回日志库同图候选（文档级向量）→ 问题级余弦复核
  3. 决策表：新分更高 → 替换（旧向量删除后重写）；同分保新；更低 → 丢弃
容量上限（dedup.max_log_records，0=自动=1.5×切片库条数）：
  超限按 (正确率升序, 时间戳升序) 淘汰，被淘汰记录同时删除向量。
语义路径异常自动降级为仅精确匹配，不阻断入库。

用法:
    python agent/data/flush_pending_logs.py                 # 默认阈值 50
    python agent/data/flush_pending_logs.py --threshold 20
"""
# Windows 下 pyarrow(pandas 间接依赖) 先加载会导致 ortools DLL 加载失败(WinError 127)。
# 必须先于 pandas 导入 k_means_constrained，让 ortools 扩展 DLL 先进入进程。
import k_means_constrained  # noqa: F401

import argparse
import os
import sys

# 确保 agent/ 与项目根在 sys.path（脚本在 agent/data/ 下运行时；utils/model 包位于项目根）
_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_AGENT_DIR)
for _p in (_PROJECT_ROOT, _AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

from model.factory import huggingface_embed_model
from rag.core.config import rag_config
from rag.core.exact_index import ExactVectorIndex
from rag.membership.log_manager import (
    LOG_COLUMNS, LogManager, merge_log_records, fetch_slices_content,
    generate_log_vector_id, dedup_in_batch, find_semantic_duplicates,
    enforce_log_cap,
)
from rag.membership.staging import count_pending_records, clear_pending_records
from rag.stores.slice_store import SliceStore
from utils.logger_handler import logger


def run_flush(pending_path, log_manager, slice_store, threshold=50,
              max_log_records=None, logs_index=None, embed_fn=None):
    """执行批量入库。返回统计 dict；pending 条数 < threshold 时不入库、不清空。

    max_log_records: 覆盖配置（测试注入）；None → rag_config.dedup_max_log_records（0=自动）
    logs_index / embed_fn: 语义去重组件注入（测试用）；None → 生产组件
    """
    pending_count = count_pending_records(pending_path)
    if pending_count < threshold:
        print(f"⏭ 暂存区记录 {pending_count} 条 < 阈值 {threshold}，跳过入库")
        return {"pending": pending_count, "added": 0, "replaced": 0,
                "dropped": 0, "flushed": False}

    df = pd.read_csv(pending_path)
    records = df.to_dict("records")

    embed_fn = embed_fn or huggingface_embed_model
    sim_threshold = rag_config.dedup_sim_threshold
    pre_filter = rag_config.dedup_pre_filter

    # 语义去重（异常降级为仅精确匹配，不阻断入库）
    in_batch_dropped = 0
    semantic_dups = {}
    try:
        # ① 批内同图高相似合并（先于切片内容补取，少做无用 get_by_ids）
        dropped_idx = dedup_in_batch(records, embed_fn, sim_threshold)
        records = [r for i, r in enumerate(records) if i not in set(dropped_idx)]
        in_batch_dropped = len(dropped_idx)

        if records:
            # ② 库内语义查重（文档级召回候选 + 问题级判定）
            if logs_index is None:
                logs_index = ExactVectorIndex(
                    collection=log_manager.logs_collection._collection,
                    persist_directory=rag_config.persist_directory,
                    collection_name=rag_config.log_collection_name,
                    embedding_fn=embed_fn,
                )
            semantic_dups = find_semantic_duplicates(
                log_manager.log_df, records, logs_index, embed_fn,
                sim_threshold, pre_filter,
            )
    except Exception as e:
        logger.warning("[flush] 语义去重失败，降级为仅精确匹配: %s", e)

    # 补切片内容
    for rec in records:
        slice_ids = str(rec.get("retrieved_slices", "")).split("|")
        rec["retrieved_slices_content"] = fetch_slices_content(slice_store, slice_ids)
        rec["correct"] = int(float(rec.get("correct", 0))) if not pd.isna(rec.get("correct")) else 0
        rec["correctness_score"] = float(rec.get("correctness_score", 0.0))

    # ③ 查重合并（精确 + 语义；replaced_keys 为实际被替换的旧行键）
    merged_df, stats, replaced_keys = merge_log_records(
        log_manager.log_df, records, semantic_dups=semantic_dups
    )
    stats["in_batch_dropped"] = in_batch_dropped

    # ④ 容量上限淘汰
    if max_log_records is None:
        max_log_records = rag_config.dedup_max_log_records
        if max_log_records == 0:
            max_log_records = round(1.5 * slice_store.count())
    merged_df, evicted_keys = enforce_log_cap(merged_df, max_log_records)

    # 写 CSV
    os.makedirs(os.path.dirname(rag_config.log_path), exist_ok=True)
    merged_df.to_csv(rag_config.log_path, index=False, encoding="utf-8")
    log_manager.log_df = merged_df

    # 向量库：先删被替换旧向量与被淘汰向量，再批量 add 全量新记录
    if evicted_keys:
        evict_ids = [generate_log_vector_id(i, q) for i, q in evicted_keys]
        try:
            log_manager.logs_collection.delete(ids=evict_ids)
            logger.info("[flush] 容量上限淘汰 %d 条并删除对应向量", len(evict_ids))
        except Exception as e:
            logger.error("[flush] 淘汰记录向量删除失败: %s", e)

    upsert_result = log_manager.upsert_records_to_vector_db(
        merged_df.to_dict("records"), replaced_keys
    )

    # 设计条款：向量库失败 → 保留 pending 不清空（CSV 已更新属可接受，下次重跑幂等）
    if upsert_result.get("added", 0) != len(merged_df):
        logger.warning(
            "[flush] 向量库 upsert 未全量成功（added=%s/%s），保留 pending 供下次重试",
            upsert_result.get("added", 0), len(merged_df),
        )
        print(f"⚠️ 向量库写入失败（{upsert_result.get('added', 0)}/{len(merged_df)}），"
              f"pending 保留待重试（CSV 已更新）")
        return {"pending": pending_count, **stats, "flushed": False}

    # 备份并清空暂存区
    backup = clear_pending_records(pending_path)
    logger.info("暂存区已清空，备份: %s", backup)

    print(f"✅ 入库完成: 新增 {stats['added']} | 替换 {stats['replaced']} | "
          f"丢弃 {stats['dropped']} | 批内合并 {stats['in_batch_dropped']} | "
          f"淘汰 {len(evicted_keys)} | 暂存 {pending_count} 条已清空")
    return {"pending": pending_count, **stats, "flushed": True}


def main():
    parser = argparse.ArgumentParser(description="隶属度日志批量入库")
    parser.add_argument("--threshold", type=int, default=50, help="待入库条数阈值")
    args = parser.parse_args()
    run_flush(rag_config.pending_log_path, LogManager(), SliceStore(),
              threshold=args.threshold)


if __name__ == "__main__":
    main()
