"""
隶属度历史日志批量入库脚本（手动运行）
检测暂存区（agent/data/pending_records.csv）待入库记录条数，
超过阈值后按 图片+问题 查重合并（新加/高分替换）并批量写入日志库与向量库，
成功后备份并清空暂存区。

用法:
    python agent/data/flush_pending_logs.py                 # 默认阈值 50
    python agent/data/flush_pending_logs.py --threshold 20
"""
import argparse
import os
import sys

_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

import pandas as pd

from rag.core.config import rag_config
from rag.membership.log_manager import (
    LOG_COLUMNS, LogManager, merge_log_records, fetch_slices_content,
)
from rag.membership.staging import count_pending_records, clear_pending_records
from rag.stores.slice_store import SliceStore
from utils.logger_handler import logger


def run_flush(pending_path, log_manager, slice_store, threshold=50):
    """执行批量入库。返回统计 dict；pending 条数 < threshold 时不入库、不清空。"""
    pending_count = count_pending_records(pending_path)
    if pending_count < threshold:
        print(f"⏭ 暂存区记录 {pending_count} 条 < 阈值 {threshold}，跳过入库")
        return {"pending": pending_count, "added": 0, "replaced": 0,
                "dropped": 0, "flushed": False}

    df = pd.read_csv(pending_path)
    records = df.to_dict("records")

    # 补切片内容
    for rec in records:
        slice_ids = str(rec.get("retrieved_slices", "")).split("|")
        rec["retrieved_slices_content"] = fetch_slices_content(slice_store, slice_ids)
        rec["correct"] = int(float(rec.get("correct", 0))) if not pd.isna(rec.get("correct")) else 0
        rec["correctness_score"] = float(rec.get("correctness_score", 0.0))

    # 查重合并（与现有日志库）；replaced_keys 为实际被替换的 (id, question) 键
    merged_df, stats, replaced_keys = merge_log_records(log_manager.log_df, records)

    # 写 CSV
    os.makedirs(os.path.dirname(rag_config.log_path), exist_ok=True)
    merged_df.to_csv(rag_config.log_path, index=False, encoding="utf-8")
    log_manager.log_df = merged_df

    # 向量库 upsert（先删被替换旧向量，再批量 add 全量新记录）
    log_manager.upsert_records_to_vector_db(merged_df.to_dict("records"), replaced_keys)

    # 备份并清空暂存区
    backup = clear_pending_records(pending_path)
    logger.info("暂存区已清空，备份: %s", backup)

    print(f"✅ 入库完成: 新增 {stats['added']} | 替换 {stats['replaced']} | "
          f"丢弃 {stats['dropped']} | 暂存 {pending_count} 条已清空")
    return {"pending": pending_count, **stats, "flushed": True}


def main():
    parser = argparse.ArgumentParser(description="隶属度日志批量入库")
    parser.add_argument("--threshold", type=int, default=50, help="待入库条数阈值")
    args = parser.parse_args()
    run_flush(rag_config.pending_log_path, LogManager(), SliceStore(),
              threshold=args.threshold)


if __name__ == "__main__":
    main()
