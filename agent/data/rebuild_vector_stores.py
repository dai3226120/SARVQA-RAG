"""
向量库全量重建脚本（嵌入模型更换后使用）

不重新生成任何源数据，仅用现有数据按新嵌入模型重新入库：
  切片库: agent/data/sar_slices.csv        → sar_slices_collection
  知识库: agent/data/*.md（3 个遥感语料）   → agent
  日志库: agent/data/rag_feedback_logs.csv → rag_test_logs

用法:
    python agent/data/rebuild_vector_stores.py               # 三个库全量重建
    python agent/data/rebuild_vector_stores.py --only slices
    python agent/data/rebuild_vector_stores.py --only knowledge
    python agent/data/rebuild_vector_stores.py --only logs
"""
# Windows 下 pyarrow(pandas 间接依赖) 先加载会导致 ortools DLL 加载失败(WinError 127)。
# 必须先于 pandas 导入 k_means_constrained，让 ortools 扩展 DLL 先进入进程。
import k_means_constrained  # noqa: F401

import argparse
import os
import shutil
import sys
import time

# Windows DLL 冲突规避（同 tests/conftest.py）：模型栈（onnxruntime/torch）DLL 先加载
# 会使 ortools 的 WinDLL 加载失败（WinError 127），rag 导入链必然触发，故在此预先加载
import ortools  # noqa: E402,F401
import k_means_constrained  # noqa: E402,F401

_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_AGENT_DIR)
for _p in (_PROJECT_ROOT, _AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

from rag.core.config import rag_config
from rag.core.chroma_manager import ChromaManager
from model.factory import huggingface_embed_model
from rag.stores.slice_store import SliceStore
from rag.membership.log_manager import LogManager
from utils.logger_handler import logger


def _clear_and_rebuild(collection_name: str) -> ChromaManager:
    """删除并重建指定 Chroma 集合（新集合自动使用新嵌入模型），返回可写入的 manager"""
    from langchain_chroma import Chroma

    mgr = ChromaManager(Chroma(
        collection_name=collection_name,
        embedding_function=huggingface_embed_model,
        persist_directory=rag_config.persist_directory,
    ))
    mgr.clear_and_rebuild_collection(
        collection_name, rag_config.persist_directory, huggingface_embed_model
    )
    return mgr


def rebuild_slices() -> dict:
    """切片库：清空集合 → 用现有 sar_slices.csv 重嵌入（不重新生成切片表）"""
    start = time.time()
    print("\n=== 重建切片库 ===")
    csv_path = rag_config.slice_csv_path
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"切片表不存在: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["slice_content"] = df["slice_content"].fillna("")
    print(f"读取切片表: {len(df)} 条")

    mgr = _clear_and_rebuild(rag_config.slices_collection_name)
    added = mgr.add_texts_in_batches(
        texts=df["slice_content"].tolist(),
        metadatas=[
            {"cluster_id": int(r["cluster_id"]), "slice_id": str(r["slice_id"])}
            for _, r in df.iterrows()
        ],
        ids=df["slice_id"].tolist(),
        desc="重建切片库",
    )
    print(f"✅ 切片库重建完成: {added}/{len(df)} 条, 耗时 {time.time()-start:.1f} 秒")
    return {"name": "slices", "total": len(df), "added": added}


def rebuild_knowledge() -> dict:
    """知识库：清空集合 + 重置 MD5 → 用 3 个 md 语料重新切分入库"""
    start = time.time()
    print("\n=== 重建知识库 ===")
    md_files = [
        "remote_sensing_imagery_corpus.md",
        "remote_sensing_rag_corpus.md",
        "SAR_remote_sensing_rag_corpus.md",
    ]
    for name in md_files:
        p = os.path.join(rag_config.data_path, name)
        if not os.path.exists(p):
            raise FileNotFoundError(f"语料文件不存在: {p}")

    # 重置 MD5 记录，强制重新入库（备份旧记录）
    if os.path.exists(rag_config.md5_store_path):
        backup = rag_config.md5_store_path + f"_backup_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.copy(rag_config.md5_store_path, backup)
        print(f"MD5 记录已备份: {backup}")
        os.remove(rag_config.md5_store_path)

    # 注意顺序：必须先清空重建集合，再创建 KnowledgeBuilder——
    # 否则其内部 store 指向被删除的旧集合，写入会报 Collection does not exist
    mgr = _clear_and_rebuild(rag_config.collection_name)

    from rag.builders.knowledge_builder import KnowledgeBuilder

    builder = KnowledgeBuilder()  # 重建后创建，store 连接新集合
    builder.load_document()
    count = mgr.count()
    print(f"✅ 知识库重建完成: 集合内 {count} 条, 耗时 {time.time()-start:.1f} 秒")
    return {"name": "knowledge", "total": count, "added": count}


def rebuild_logs() -> dict:
    """日志库：保留 CSV，清空集合 → 用现有 rag_feedback_logs.csv 重嵌入（不重新生成）"""
    start = time.time()
    print("\n=== 重建日志库 ===")
    csv_path = rag_config.log_path
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"日志文件不存在: {csv_path}")

    # clear_vector_db_only: 保留 CSV，仅清空向量集合；reload 用新模型全量重嵌入
    log_manager = LogManager(clear_vector_db_only=True)
    total = len(log_manager.log_df)
    print(f"读取日志 CSV: {total} 条")
    log_manager.reload_to_vector_db()
    print(f"✅ 日志库重建完成: {total} 条, 耗时 {time.time()-start:.1f} 秒")
    return {"name": "logs", "total": total, "added": total}


_BUILDERS = {
    "slices": rebuild_slices,
    "knowledge": rebuild_knowledge,
    "logs": rebuild_logs,
}


def main():
    parser = argparse.ArgumentParser(description="向量库全量重建（嵌入模型更换后使用）")
    parser.add_argument("--only", choices=list(_BUILDERS.keys()),
                        help="只重建指定库（默认全部）")
    args = parser.parse_args()

    targets = [args.only] if args.only else list(_BUILDERS.keys())
    print(f"待重建: {', '.join(targets)}（源数据不重新生成，仅重新嵌入）")

    results = []
    for name in targets:
        try:
            results.append(_BUILDERS[name]())
        except Exception as e:
            print(f"❌ {name} 重建失败: {e}")
            logger.error("重建失败: %s", e, exc_info=True)

    print("\n" + "=" * 50)
    for r in results:
        print(f"✅ {r['name']}: {r['added']}/{r['total']} 条")
    if len(results) < len(targets):
        print("⚠️ 部分库重建失败，请检查上方日志")
    print("重建完成！请重启在线服务，并建议重跑一次 init_membership_logs.py --force")


if __name__ == "__main__":
    main()
