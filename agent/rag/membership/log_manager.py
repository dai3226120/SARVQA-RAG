"""
日志向量库管理模块
源自原 rag_rsfit_builder.py SARSemanticCacheSystem 中：
  - __init__ 中的日志加载/增量逻辑
  - _deduplicate_log_df / _check_and_reload_vector_db / _get_vector_db_record_count
  - _incrementally_add_new_records / _reload_logs_to_vector_db
  - _generate_unique_id / _is_md5_registered / _append_md5_record
"""
import os
import hashlib
from datetime import datetime

import pandas as pd
from langchain_chroma import Chroma

from rag.core.config import rag_config
from rag.core.md5_store import Md5Store
from rag.core.chroma_manager import ChromaManager
from model.factory import huggingface_embed_model
from utils.file_handler import get_file_md5_hex
from utils.logger_handler import logger


class LogManager:
    """
    日志向量库生命周期管理器
    负责：日志 CSV 加载/去重/增量同步/全量重载/MD5 校验
    """

    def __init__(
        self,
        force_full_reload: bool = False,
        clear_vector_db_only: bool = False,
    ):
        """
        Args:
            force_full_reload: 强制全量重载 - 清空 CSV + 向量库
            clear_vector_db_only: 只清除向量库 - 保留 CSV
        """
        self._embeddings = huggingface_embed_model
        self._persist_directory = rag_config.persist_directory
        self._md5_store = Md5Store(rag_config.md5_log_store_path)

        # 初始化日志向量库
        self._logs_collection = Chroma(
            collection_name=rag_config.log_collection_name,
            embedding_function=self._embeddings,
            persist_directory=self._persist_directory,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self._chroma_mgr = ChromaManager(self._logs_collection)

        os.makedirs(rag_config.data_path, exist_ok=True)

        # 根据模式初始化日志
        if force_full_reload:
            self._init_force_reload()
        elif clear_vector_db_only:
            self._init_clear_vector_db()
        elif os.path.exists(rag_config.log_path):
            self._init_incremental()
        else:
            self.log_df = pd.DataFrame(
                columns=[
                    "id",
                    "question",
                    "retrieved_slices",
                    "correctness_score",
                    "predicted_text",
                    "timestamp",
                ]
            )

    @property
    def log_df(self) -> pd.DataFrame:
        return self._log_df

    @log_df.setter
    def log_df(self, value: pd.DataFrame):
        self._log_df = value

    @property
    def logs_collection(self):
        return self._logs_collection

    # ── 初始化模式 ──

    def _init_force_reload(self):
        """模式1: 强制全量重载"""
        logger.warning("[强制全量更新] force_full_reload=True，清空所有日志并重建...")

        self._log_df = pd.DataFrame(
            columns=[
                "id",
                "question",
                "retrieved_slices",
                "correctness_score",
                "predicted_text",
                "timestamp",
            ]
        )
        self._save_log_df()
        logger.info(f"[强制全量更新] 已清空反馈日志文件: {rag_config.log_path}")

        error_df = pd.DataFrame(
            columns=[
                "id",
                "question",
                "retrieved_slices",
                "correctness_score",
                "predicted_text",
                "ground_truth",
                "llm_judge",
                "timestamp",
            ]
        )
        error_df.to_csv(rag_config.error_log_path, index=False, encoding="utf-8")

        self.reload_to_vector_db()
        logger.info("[强制全量更新] 向量库已清空并重建完成")

    def _init_clear_vector_db(self):
        """模式2: 只清除向量库"""
        logger.warning("[仅清空向量库] clear_vector_db_only=True，保留CSV，清除向量库...")

        if os.path.exists(rag_config.log_path):
            self._log_df = pd.read_csv(rag_config.log_path)
            for col in [
                "id",
                "question",
                "retrieved_slices",
                "correctness_score",
                "predicted_text",
                "timestamp",
            ]:
                if col not in self._log_df.columns:
                    self._log_df[col] = ""
            self._deduplicate_log_df()
            logger.info(f"[仅清空向量库] 已加载 {len(self._log_df)} 条CSV记录")
        else:
            self._log_df = pd.DataFrame(
                columns=[
                    "id",
                    "question",
                    "retrieved_slices",
                    "correctness_score",
                    "predicted_text",
                    "timestamp",
                ]
            )

        self._chroma_mgr.clear_and_rebuild_collection(
            collection_name=rag_config.log_collection_name,
            persist_directory=self._persist_directory,
            embedding_function=self._embeddings,
            space="cosine",
        )

    def _init_incremental(self):
        """模式3: 增量更新（默认）"""
        self._log_df = pd.read_csv(rag_config.log_path)
        for col in [
            "id",
            "question",
            "retrieved_slices",
            "correctness_score",
            "predicted_text",
            "timestamp",
        ]:
            if col not in self._log_df.columns:
                self._log_df[col] = ""

        self._deduplicate_log_df()
        self._check_and_reload_vector_db()

    # ── 日志操作 ──

    def _deduplicate_log_df(self):
        """对日志数据进行去重，保留最新的记录"""
        if self._log_df.empty:
            return

        original_length = len(self._log_df)
        self._log_df["timestamp"] = pd.to_datetime(
            self._log_df["timestamp"], errors="coerce"
        )
        self._log_df = self._log_df.sort_values("timestamp", ascending=False)
        self._log_df = self._log_df.drop_duplicates(subset=["id"], keep="first")
        self._log_df = self._log_df.reset_index(drop=True)

        self._save_log_df()

        if len(self._log_df) < original_length:
            logger.info(
                f"日志数据去重完成，移除 {original_length - len(self._log_df)} 条重复记录，"
                f"当前有效记录数：{len(self._log_df)}"
            )
        else:
            logger.info(
                f"日志数据去重完成，当前有效记录数：{len(self._log_df)}（无重复数据）"
            )

    def _save_log_df(self):
        """保存日志 DataFrame 到 CSV"""
        os.makedirs(os.path.dirname(rag_config.log_path), exist_ok=True)
        self._log_df.to_csv(rag_config.log_path, index=False, encoding="utf-8")

    def _check_and_reload_vector_db(self):
        """检查 CSV 记录数量，只有增加时才增量添加"""
        try:
            csv_count = len(self._log_df)
            db_count = self._chroma_mgr.count()

            logger.info(f"[增量检查] CSV记录数: {csv_count}, 向量库记录数: {db_count}")

            if csv_count > db_count:
                new_count = csv_count - db_count
                logger.warning(
                    f"[增量添加] 检测到有 {new_count} 条新记录，开始增量添加..."
                )
                self._incrementally_add_new_records()
            else:
                logger.info("[增量检查] 向量库记录数已同步，无需添加新记录。")
        except Exception as e:
            logger.error("增量检查失败: %s", e, exc_info=True)
            logger.warning("[降级策略] 增量检查失败，尝试全量重载...")
            self.reload_to_vector_db()

    def _incrementally_add_new_records(self):
        """增量添加新记录到向量库"""
        if self._log_df.empty:
            logger.info("[增量添加] CSV日志为空，无需添加。")
            return

        try:
            existing_count = self._chroma_mgr.count()

            if existing_count >= len(self._log_df):
                logger.info("[增量添加] 向量库记录数已超过CSV，无需添加新记录。")
                return

            new_records = self._log_df.iloc[existing_count:]
            logger.info(
                f"[增量添加] 准备添加 {len(new_records)} 条新记录..."
            )

            texts, metadatas, ids = self._build_log_texts(new_records)
            self._chroma_mgr.add_texts_in_batches(
                texts=texts, metadatas=metadatas, ids=ids, desc="增量添加日志"
            )

            final_count = self._chroma_mgr.count()
            logger.info(
                f"[增量添加] 完成！向量库现有 {final_count} 条记录。"
            )
        except Exception as e:
            logger.error("[增量添加] 失败: %s", e, exc_info=True)
            logger.warning("[增量添加] 降级为全量重载...")
            self.reload_to_vector_db()

    def reload_to_vector_db(self):
        """清空向量集合并全量重载"""
        logger.info("[reload_to_vector_db] 开始执行重载...")

        if self._log_df.empty:
            logger.info("CSV 日志为空，无需同步至向量数据库。")
            return

        self._chroma_mgr.clear_and_rebuild_collection(
            collection_name=rag_config.log_collection_name,
            persist_directory=self._persist_directory,
            embedding_function=self._embeddings,
            space="cosine",
        )

        texts, metadatas, ids = self._build_log_texts(self._log_df)
        self._chroma_mgr.add_texts_in_batches(
            texts=texts, metadatas=metadatas, ids=ids, desc="全量重载日志"
        )

        logger.info(f"重载完成，成功全量同步 {len(texts)} 条记录至日志向量库。")

    def add_single_record(self, log_entry: dict):
        """添加单条记录到日志和向量库"""
        doc_text = (
            f"Question: {log_entry['question']}\n"
            f"Retrieved Slices: {log_entry['retrieved_slices']}"
        )
        unique_id = self._generate_unique_id(str(log_entry["id"]))

        self._logs_collection.add_texts(
            texts=[doc_text],
            metadatas=[{
                "id": str(log_entry["id"]),
                "question": log_entry["question"],
                "retrieved_slices": log_entry["retrieved_slices"],
                "correctness_score": str(log_entry["correctness_score"]),
                "predicted_text": log_entry["predicted_text"],
                "timestamp": log_entry["timestamp"],
            }],
            ids=[unique_id],
        )

    def update_md5(self):
        """更新日志文件的 MD5 记录"""
        try:
            new_md5 = get_file_md5_hex(rag_config.log_path)
            if new_md5 and not self._md5_store.is_registered(new_md5):
                self._md5_store.register(new_md5)
                logger.info(f"新 MD5 已记录: {new_md5}")
        except Exception as e:
            logger.error(f"更新 MD5 失败: {e}")

    # ── 辅助方法 ──

    def _build_log_texts(self, df: pd.DataFrame) -> tuple:
        """从 DataFrame 构建向量库写入所需的 texts/metadatas/ids"""
        texts = []
        metadatas = []
        ids = []

        for idx, (_, row) in enumerate(df.iterrows()):
            if pd.isna(row["question"]) or str(row["id"]).strip() == "":
                continue

            doc_text = (
                f"Question: {row['question']}\n"
                f"Retrieved Slices: {row['retrieved_slices']}"
            )
            texts.append(doc_text)
            metadatas.append({
                "id": str(row["id"]),
                "question": str(row["question"]),
                "retrieved_slices": (
                    str(row["retrieved_slices"]) if not pd.isna(row["retrieved_slices"]) else ""
                ),
                "correctness_score": (
                    str(row["correctness_score"]) if not pd.isna(row["correctness_score"]) else "0.0"
                ),
                "predicted_text": (
                    str(row["predicted_text"]) if not pd.isna(row["predicted_text"]) else ""
                ),
                "timestamp": (
                    str(row["timestamp"]) if not pd.isna(row["timestamp"]) else ""
                ),
            })
            ids.append(self._generate_unique_id(str(row["id"])))

        return texts, metadatas, ids

    @staticmethod
    def _generate_unique_id(base_id: str) -> str:
        """生成唯一的向量库 ID"""
        unique_suffix = hashlib.md5(
            f"{base_id}_{datetime.now().timestamp()}".encode()
        ).hexdigest()[:8]
        return f"log_{base_id}_{unique_suffix}"
