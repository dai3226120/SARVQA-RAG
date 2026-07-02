"""
SAR 切片构建器
负责 SAR-VQA CSV 数据的聚类、切片生成和向量库入库
源自原 rag_rscsv_builder.py (RscsvBuilder)
"""
import os
import pandas as pd
import numpy as np
from k_means_constrained import KMeansConstrained
from tqdm import tqdm
from langchain_chroma import Chroma

from rag.stores.slice_store import SliceStore
from rag.core.config import rag_config
from rag.core.md5_store import Md5Store
from rag.core.chroma_manager import ChromaManager
from model.factory import huggingface_embed_model
from utils.file_handler import get_file_md5_hex
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


class SliceBuilder:
    """SAR 数据切片构建器，负责聚类、切片生成和向量库入库"""

    def __init__(self):
        self._store = SliceStore()
        self._md5_store = Md5Store(rag_config.md5_store_path)
        self._chroma_mgr = ChromaManager(self._store.collection)

    @property
    def store(self) -> SliceStore:
        return self._store

    def get_retriever(self, k: int = None):
        """获取向量检索器"""
        k = k or rag_config.k
        return self._store.get_retriever(k=k)

    def _ensure_directories(self):
        """确保所有必要目录存在"""
        if not os.path.exists(rag_config.persist_directory):
            os.makedirs(rag_config.persist_directory, exist_ok=True)
            logger.info(f"[SliceBuilder] {rag_config.persist_directory} 目录不存在，已创建。")

    def build_sar_slice(self, csv_path: str = None, output_dir: str = None):
        """
        构建 SAR 切片表

        Args:
            csv_path: 输入 CSV 路径，默认使用配置路径
            output_dir: 输出目录，默认使用配置路径

        Returns:
            切片 DataFrame
        """
        self._ensure_directories()
        csv_path = csv_path or rag_config.sar_csv_path
        output_dir = output_dir or rag_config.data_path

        if not os.path.exists(csv_path):
            logger.info(f"[SliceBuilder] SAR 数据文件不存在: {csv_path}。")
            raise FileNotFoundError(f"SAR 数据文件不存在: {csv_path}")

        # MD5 增量检查
        slice_out_path = os.path.join(output_dir, "sar_slices.csv")
        csv_md5 = get_file_md5_hex(csv_path)
        slice_md5 = get_file_md5_hex(slice_out_path)

        if (
            csv_md5
            and slice_md5
            and os.path.exists(slice_out_path)
            and self._md5_store.is_registered(csv_md5)
            and self._md5_store.is_registered(slice_md5)
        ):
            logger.info(f"[SliceBuilder] SAR CSV 未发生变化，跳过重构。")
            return pd.read_csv(slice_out_path, encoding="utf-8-sig")

        # 数据加载与预处理
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if "question" not in df.columns or "answer" not in df.columns:
            raise ValueError("SAR CSV 必须包含 'question' 和 'answer' 列")

        df["answer"] = df["answer"].fillna("")
        df["text"] = df["question"].astype(str) + " " + df["answer"].astype(str)

        # 嵌入生成
        logger.info(f"[SliceBuilder] 正在生成文本语义嵌入...")
        embeddings = self._embed_documents(df["text"].tolist())

        # 聚类
        df["cluster_id"] = self._cluster(embeddings, len(df))

        # 生成切片记录
        slice_records = self._generate_slices(df)

        # 保存
        slice_df = pd.DataFrame(slice_records)
        slice_df.to_csv(slice_out_path, index=False, encoding="utf-8-sig")
        logger.info(f"[SliceBuilder] 已生成切片表: {slice_out_path}，共 {len(slice_df)} 条")

        if csv_md5:
            self._md5_store.register(csv_md5)

        return slice_df

    def build_chroma_collections(self, csv_path: str = None, output_dir: str = None):
        """
        构建 Chroma 向量数据库集合

        Args:
            csv_path: 输入 CSV 路径
            output_dir: 输出目录

        Returns:
            切片向量库集合
        """
        self._ensure_directories()
        csv_path = csv_path or rag_config.sar_csv_path
        output_dir = output_dir or rag_config.data_path

        if not os.path.exists(csv_path):
            logger.info(f"[SliceBuilder] SAR 数据文件不存在: {csv_path}。")
            raise FileNotFoundError(f"SAR 数据文件不存在: {csv_path}")

        slice_out_path = os.path.join(output_dir, "sar_slices.csv")
        csv_md5 = get_file_md5_hex(csv_path)
        slice_md5 = get_file_md5_hex(slice_out_path)

        if (
            csv_md5
            and slice_md5
            and os.path.exists(slice_out_path)
            and self._md5_store.is_registered(csv_md5)
            and self._md5_store.is_registered(slice_md5)
        ):
            logger.info(f"[SliceBuilder] SAR CSV 未发生变化，跳过重构。")
            return self._store

        slice_file = os.path.join(rag_config.data_path, "sar_slices.csv")
        if not os.path.exists(slice_file):
            raise FileNotFoundError(f"切片表不存在: {slice_file}")

        slice_df = pd.read_csv(slice_file, encoding="utf-8-sig")
        logger.info(f"[SliceBuilder] 正在初始化 Chroma Embeddings...")

        # 增量写入
        existing_ids = self._chroma_mgr.get_existing_ids()
        logger.info(f"[SliceBuilder] 基础切片库已有 {len(existing_ids)} 条记录")

        new_slices = slice_df[~slice_df["slice_id"].isin(existing_ids)]
        if not new_slices.empty:
            logger.info(f"[SliceBuilder] 需增量写入 {len(new_slices)} 条记录")

            slice_texts = new_slices["slice_content"].tolist()
            slice_metadatas = [
                {"cluster_id": int(row["cluster_id"]), "slice_id": str(row["slice_id"])}
                for _, row in new_slices.iterrows()
            ]
            slice_ids = new_slices["slice_id"].tolist()

            self._chroma_mgr.add_texts_in_batches(
                texts=slice_texts,
                metadatas=slice_metadatas,
                ids=slice_ids,
                desc="写入SAR切片",
            )
        else:
            logger.info(f"[SliceBuilder] 无需增量写入基础切片库")

        slice_md5 = get_file_md5_hex(slice_out_path)
        if slice_md5:
            self._md5_store.register(slice_md5)

        return self._store

    def build_all(self) -> None:
        """执行完整构建流程"""
        self.build_chroma_collections()

        # 外部 MD5 检查，确保增量处理正确记录
        slice_out_path = os.path.join(rag_config.data_path, "sar_slices.csv")
        logger.info(f"slice_out_path: {slice_out_path}")
        slice_md5 = get_file_md5_hex(slice_out_path)
        logger.info(f"slice_md5: {slice_md5}")
        if slice_md5:
            self._md5_store.register(slice_md5)

    # ── 私有辅助方法 ──

    def _embed_documents(self, texts: list[str]) -> list:
        """批量生成嵌入向量"""
        return huggingface_embed_model.embed_documents(texts)

    def _cluster(self, embeddings: list, n_samples: int) -> list:
        """执行带约束的 KMeans 聚类"""
        n_clusters = rag_config.n_clusters
        slice_size = rag_config.slice_size

        if n_clusters <= 0:
            n_clusters = 1

        base_size = max(1, n_samples // n_clusters)
        remainder = n_samples % n_clusters
        size_min = base_size
        size_max = base_size + 1 if remainder else base_size
        size_max = max(size_max, size_min)

        logger.info(
            f"[SliceBuilder] 目标簇数: {n_clusters}, 簇大小范围: {size_min}~{size_max}"
        )

        clf = KMeansConstrained(
            n_clusters=n_clusters,
            size_min=size_min,
            size_max=size_max,
            random_state=42,
            n_init=10,
            n_jobs=1,
        )
        return clf.fit_predict(embeddings)

    def _generate_slices(self, df: pd.DataFrame) -> list[dict]:
        """从聚类结果生成切片记录"""
        slice_size = rag_config.slice_size
        slice_records = []

        for cid in range(rag_config.n_clusters):
            cluster_df = df[df["cluster_id"] == cid]
            if cluster_df.empty:
                continue

            cluster_indices = cluster_df.index.tolist()
            for start in range(0, len(cluster_indices), slice_size):
                slice_idx = cluster_indices[start : start + slice_size]
                slice_data = cluster_df.loc[slice_idx]

                slice_content = "\n".join(
                    [
                        f"Q: {row['question']} A: {row['answer']}"
                        for _, row in slice_data.iterrows()
                    ]
                )

                slice_records.append({
                    "slice_id": f"sar_cluster_{cid}_slice_{start // slice_size}",
                    "cluster_id": cid,
                    "slice_content": slice_content,
                })

        return slice_records
