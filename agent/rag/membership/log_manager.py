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

import pandas as pd
from langchain_chroma import Chroma

from rag.core.config import rag_config
from rag.core.md5_store import Md5Store
from rag.core.chroma_manager import ChromaManager
from model.factory import huggingface_embed_model
from utils.file_handler import get_file_md5_hex
from utils.logger_handler import logger


# ====================== 新 schema 常量与查重/替换纯逻辑 ======================
LOG_COLUMNS = [
    "id", "question", "retrieved_slices", "retrieved_slices_content",
    "predicted_text", "correct", "correctness_score", "timestamp",
]


def record_key(id_: str, question: str) -> str:
    """图片+问题 联合查重键"""
    return f"{id_}__{question}"


def generate_log_vector_id(base_id: str, question: str) -> str:
    """确定性向量库 id：同一 (图片, 问题) 记录始终映射到同一向量 id，替换时可定位删除"""
    q_hash = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
    return f"log_{base_id}_{q_hash}"


def merge_log_records(existing_df: pd.DataFrame, new_records: list[dict]):
    """按 图片+问题 查重合并日志记录。

    决策表：
      - 键不存在        → added（追加）
      - 存在且新分数更高 → replaced（整体替换，时间戳用新的）
      - 存在且分数相等   → 保留时间戳更晚的一条（同分保新）
      - 存在且新分数更低 → dropped（保留旧值）

    Returns:
        (merged_df, {"added": int, "replaced": int, "dropped": int}, replaced_keys)
        replaced_keys: 实际发生替换的 (id, question) 键列表，供向量库定位删除旧向量
    """
    df = existing_df.copy()
    if df.empty:
        df = pd.DataFrame(columns=LOG_COLUMNS)
    for col in LOG_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[LOG_COLUMNS]

    stats = {"added": 0, "replaced": 0, "dropped": 0}
    replaced_keys = []
    key_to_idx = {record_key(r["id"], r["question"]): i for i, r in df.iterrows()}

    for rec in new_records:
        key = record_key(rec["id"], rec["question"])
        new_score = float(rec.get("correctness_score", 0.0))
        new_ts = str(rec.get("timestamp", ""))

        if key not in key_to_idx:
            stats["added"] += 1
            row = pd.DataFrame([{c: rec.get(c, "") for c in LOG_COLUMNS}], columns=LOG_COLUMNS)
            df = pd.concat([df, row], ignore_index=True)
            key_to_idx[key] = len(df) - 1
            continue

        idx = key_to_idx[key]
        old_score = float(df.at[idx, "correctness_score"] or 0.0)
        old_ts = str(df.at[idx, "timestamp"] or "")

        if new_score > old_score:
            stats["replaced"] += 1
            replaced_keys.append((rec["id"], rec["question"]))
            for c in LOG_COLUMNS:
                df.at[idx, c] = rec.get(c, "")
        elif new_score == old_score and new_ts > old_ts:
            stats["replaced"] += 1
            replaced_keys.append((rec["id"], rec["question"]))
            for c in LOG_COLUMNS:
                df.at[idx, c] = rec.get(c, "")
        else:
            stats["dropped"] += 1

    return df, stats, replaced_keys


def build_log_doc_text(
    question: str,
    slices_content: list[str],
    answer: str,
    slice_char_limit: int = 200,
    total_char_limit: int = 1500,
) -> str:
    """构建隶属度相似度的综合向量文本：问题 + 切片内容 + 回答。

    切片内容每片截断至 slice_char_limit 字符；整体截断至 total_char_limit 字符，
    防止超出嵌入模型的输入上限。
    """
    truncated_slices = [s[:slice_char_limit] for s in slices_content if s]
    slices_part = "\n".join(truncated_slices)
    doc = f"Question: {question}\nRetrieved Slices: {slices_part}\nAnswer: {answer}"
    return doc[:total_char_limit]


def fetch_slices_content(slice_store, slices_id_list: list[str]) -> str:
    """从切片库按 id 批量取切片内容，换行拼接（缺失切片跳过）"""
    valid = [s for s in slices_id_list if s and str(s).strip()]
    if not valid:
        return ""
    result = slice_store.get_by_ids(valid)
    documents = result.get("documents", []) or []
    return "\n".join(d for d in documents if d)


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
            self.log_df = pd.DataFrame(columns=LOG_COLUMNS)

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

        self._log_df = pd.DataFrame(columns=LOG_COLUMNS)
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
            for col in LOG_COLUMNS:
                if col not in self._log_df.columns:
                    self._log_df[col] = ""
            self._deduplicate_log_df()
            logger.info(f"[仅清空向量库] 已加载 {len(self._log_df)} 条CSV记录")
        else:
            self._log_df = pd.DataFrame(columns=LOG_COLUMNS)

        self._chroma_mgr.clear_and_rebuild_collection(
            collection_name=rag_config.log_collection_name,
            persist_directory=self._persist_directory,
            embedding_function=self._embeddings,
            space="cosine",
        )

    def _init_incremental(self):
        """模式3: 增量更新（默认）"""
        self._log_df = pd.read_csv(rag_config.log_path)
        for col in LOG_COLUMNS:
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
        # 查重键 = 图片+问题 联合键（同图不同问题需各自保留，同键保留最新一条）
        self._log_df = self._log_df.drop_duplicates(
            subset=["id", "question"], keep="first"
        )
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
        """添加单条记录到日志和向量库（综合向量：问题+切片内容+回答）"""
        slices_content = str(log_entry.get("retrieved_slices_content", "")).splitlines()
        doc_text = build_log_doc_text(
            log_entry["question"], slices_content, log_entry.get("predicted_text", "")
        )
        unique_id = generate_log_vector_id(str(log_entry["id"]), log_entry["question"])

        self._logs_collection.add_texts(
            texts=[doc_text],
            metadatas=[{
                "id": str(log_entry["id"]),
                "question": log_entry["question"],
                "retrieved_slices": log_entry.get("retrieved_slices", ""),
                "retrieved_slices_content": log_entry.get("retrieved_slices_content", ""),
                "correctness_score": str(log_entry.get("correctness_score", 0.0)),
                "correct": str(log_entry.get("correct", 0)),
                "predicted_text": log_entry.get("predicted_text", ""),
                "timestamp": log_entry.get("timestamp", ""),
            }],
            ids=[unique_id],
        )

    def upsert_records_to_vector_db(
        self,
        records: list[dict],
        replaced_keys: list[tuple[str, str]] = None,
    ) -> dict:
        """批量 upsert 记录到向量库。

        Args:
            records: 合并后需入库的全部记录（含新增与替换后的新内容）
            replaced_keys: 被替换记录的 (图片, 问题) 键列表——先删旧向量再写新向量

        Returns:
            {"added": int, "deleted": int}
        """
        replaced_keys = replaced_keys or []
        deleted = 0
        if replaced_keys:
            old_ids = [
                generate_log_vector_id(id_, q) for id_, q in replaced_keys
            ]
            try:
                self._logs_collection.delete(ids=old_ids)
                deleted = len(old_ids)
            except Exception as e:
                logger.error("[upsert] 删除旧向量失败: %s", e)

        if not records:
            return {"added": 0, "deleted": deleted}

        texts, metadatas, ids = self._build_log_texts(pd.DataFrame(records, columns=LOG_COLUMNS))
        added = self._chroma_mgr.add_texts_in_batches(
            texts=texts, metadatas=metadatas, ids=ids, desc="批量 upsert 日志"
        )
        return {"added": added, "deleted": deleted}

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
            if (
                pd.isna(row["question"])
                or not str(row["question"]).strip()
                or str(row["id"]).strip() == ""
            ):
                continue

            slices_content = (
                str(row["retrieved_slices_content"]).splitlines()
                if not pd.isna(row["retrieved_slices_content"]) else []
            )
            doc_text = build_log_doc_text(
                str(row["question"]), slices_content, str(row["predicted_text"])
            )
            texts.append(doc_text)
            metadatas.append({
                "id": str(row["id"]),
                "question": str(row["question"]),
                "retrieved_slices": (
                    str(row["retrieved_slices"]) if not pd.isna(row["retrieved_slices"]) else ""
                ),
                "retrieved_slices_content": (
                    str(row["retrieved_slices_content"])
                    if not pd.isna(row["retrieved_slices_content"]) else ""
                ),
                "correctness_score": (
                    str(row["correctness_score"]) if not pd.isna(row["correctness_score"]) else "0.0"
                ),
                "correct": str(row["correct"]) if not pd.isna(row["correct"]) else "0",
                "predicted_text": (
                    str(row["predicted_text"]) if not pd.isna(row["predicted_text"]) else ""
                ),
                "timestamp": (
                    str(row["timestamp"]) if not pd.isna(row["timestamp"]) else ""
                ),
            })
            ids.append(generate_log_vector_id(str(row["id"]), str(row["question"])))

        return texts, metadatas, ids
