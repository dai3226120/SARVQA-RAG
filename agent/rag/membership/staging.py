"""
运行期新记录暂存区管理
在线系统每会话产生一条记录 → append_pending_record() 追加到 pending_records.csv
批量入库脚本（agent/data/flush_pending_logs.py）检测条数 → 超阈值 → 合并入库 → 清空
"""
import os
import shutil
from datetime import datetime

import pandas as pd

from rag.core.config import rag_config
from utils.thread_lock import lock

PENDING_COLUMNS = [
    "id", "question", "retrieved_slices", "predicted_text",
    "correct", "correctness_score", "timestamp",
]


def _resolve_path(pending_path: str = None) -> str:
    return pending_path or rag_config.pending_log_path


def append_pending_record(record: dict, pending_path: str = None) -> None:
    """线程安全地追加一条暂存记录（在线系统每次会话调用）"""
    path = _resolve_path(pending_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = {c: record.get(c, "") for c in PENDING_COLUMNS}
    with lock:
        if not os.path.exists(path):
            pd.DataFrame(columns=PENDING_COLUMNS).to_csv(path, index=False, encoding="utf-8")
        pd.DataFrame([row]).to_csv(path, index=False, mode="a", header=False, encoding="utf-8")


def count_pending_records(pending_path: str = None) -> int:
    """检测暂存区待入库记录条数（文件缺失返回 0）"""
    path = _resolve_path(pending_path)
    if not os.path.exists(path):
        return 0
    try:
        df = pd.read_csv(path)
        return len(df)
    except Exception:
        return 0


def clear_pending_records(pending_path: str = None, backup: bool = True):
    """清空暂存区（仅保留表头）；backup=True 时先备份为 *_backup_<时间戳>.csv

    Returns:
        str | None: 备份文件路径（未备份或清空失败时 None）
    """
    path = _resolve_path(pending_path)
    if not os.path.exists(path):
        return None
    backup_path = None
    if backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.replace(".csv", f"_backup_{ts}.csv")
        shutil.copy(path, backup_path)
    with lock:
        pd.DataFrame(columns=PENDING_COLUMNS).to_csv(path, index=False, encoding="utf-8")
    return backup_path
