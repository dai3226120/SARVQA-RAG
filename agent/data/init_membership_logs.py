"""
隶属度历史日志初始化脚本（手动运行）
从数据集重跑预测 → LLM 判断 → 只保留预测正确的记录 → 综合向量全量入库

用法:
    python agent/data/init_membership_logs.py                    # 默认 test.csv, 20000 条
    python agent/data/init_membership_logs.py --csv agent/dataset_split/test.csv --max-rows 5000 --force
"""
import argparse
import os
import shutil
import sys
import time
from datetime import datetime

# 确保 agent/ 与项目根在 sys.path（脚本在 agent/data/ 下运行时；utils 包位于项目根）
_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_AGENT_DIR)
for _p in (_PROJECT_ROOT, _AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

from rag.core.config import rag_config
from rag.membership.log_manager import (
    LOG_COLUMNS, LogManager, merge_log_records, fetch_slices_content,
)
from rag.membership.vlm_evaluator import VlmEvaluator
from rag.membership.llm_judge import LlmJudge
from rag.stores.slice_store import SliceStore
from utils.logger_handler import logger


def run_init(csv_path, max_rows, evaluator, judge, slice_store, log_manager, force=False):
    """执行初始化。evaluator/judge/slice_store 可注入 fake 用于测试。"""
    if not force:
        answer = input(f"将重建隶属度日志库（备份后清空重建），确认继续? [y/N]: ")
        if answer.strip().lower() != "y":
            print("已取消")
            return None

    start = time.time()
    df = pd.read_csv(csv_path, nrows=max_rows)
    logger.info("读取数据集 %s，共 %d 条", csv_path, len(df))

    # 1. 备份旧日志
    if os.path.exists(rag_config.log_path):
        backup = rag_config.log_path.replace(".csv", f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        shutil.copy(rag_config.log_path, backup)
        logger.info("旧日志已备份到 %s", backup)

    # 2. 逐条预测
    logs = []
    for _, row in df.iterrows():
        try:
            score, text, slice_ids = evaluator.call_vlm_agent(
                row["image"], row["question"], row.get("answer")
            )
            logs.append({
                "id": str(row["id"]),
                "question": str(row["question"]),
                "retrieved_slices": "|".join(slice_ids or []),
                "predicted_text": text,
                "correctness_score": score,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
        except Exception as e:
            logger.error("预测失败 [%s] %s: %s", row.get("id"), row.get("question"), e)

    # 3. LLM 判断正确性，只保留正确的
    correct_logs, error_logs = judge.judge_and_filter(df, logs)
    logger.info("LLM 判断: 正确 %d / 错误 %d", len(correct_logs), len(error_logs))

    # 4. 补切片内容 + correct 列 → 查重合并
    for log in correct_logs:
        log["correct"] = 1
        slice_ids = str(log.get("retrieved_slices", "")).split("|")
        log["retrieved_slices_content"] = fetch_slices_content(slice_store, slice_ids)
    merged_df, stats, _ = merge_log_records(
        pd.DataFrame(columns=LOG_COLUMNS), correct_logs
    )

    # 5. 写 CSV
    os.makedirs(os.path.dirname(rag_config.log_path), exist_ok=True)
    merged_df.to_csv(rag_config.log_path, index=False, encoding="utf-8")
    log_manager.log_df = merged_df

    # 6. 重建向量库
    log_manager.reload_to_vector_db()

    elapsed = time.time() - start
    logger.info("初始化完成: 预测 %d | 正确 %d | 入库 %d | 耗时 %.1f 秒",
                len(logs), len(correct_logs), len(merged_df), elapsed)
    print(f"✅ 初始化完成: 预测 {len(logs)} | 正确 {len(correct_logs)} | "
          f"入库 {len(merged_df)} | 耗时 {elapsed:.1f} 秒")
    return {"total": len(logs), "correct_count": len(correct_logs),
            "final_count": len(merged_df), "elapsed_sec": elapsed}


def main():
    parser = argparse.ArgumentParser(description="隶属度日志初始化：只保留预测正确的会话记录")
    parser.add_argument("--csv", default="agent/dataset_split/test.csv")
    parser.add_argument("--max-rows", type=int, default=20000)
    parser.add_argument("--force", action="store_true", help="跳过确认，直接备份并重建")
    args = parser.parse_args()

    run_init(
        csv_path=args.csv,
        max_rows=args.max_rows,
        evaluator=VlmEvaluator(),
        judge=LlmJudge(),
        slice_store=SliceStore(),
        log_manager=LogManager(force_full_reload=True),
        force=args.force,
    )


if __name__ == "__main__":
    main()
