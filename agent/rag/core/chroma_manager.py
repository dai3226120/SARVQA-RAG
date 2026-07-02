"""
Chroma 向量数据库统一管理模块
封装增删/计数/分批写入/重建/清空等通用操作
合并自原 vector_store.py / rag_rscsv_builder.py / rag_rsfit_builder.py 三处各异实现
"""
from abc import ABC, abstractmethod
from typing import Optional
from utils.logger_handler import logger


class ChromaManager:
    """
    Chroma 向量库的通用生命周期管理器
    提供分批写入、记录计数、集合重建、集合清空等通用操作
    所有参数通过构造函数注入，不持有持久化状态
    """

    def __init__(self, collection, batch_size: int = 500):
        """
        Args:
            collection: LangChain Chroma 集合实例
            batch_size: 分批操作时的默认批次大小
        """
        self._collection = collection
        self._batch_size = batch_size

    @property
    def collection(self):
        return self._collection

    def count(self) -> int:
        """
        获取向量库中的记录总数
        Chroma 无直接 count() 方法，通过分页 get 实现

        Returns:
            记录总数
        """
        total = 0
        offset = 0
        try:
            while True:
                result = self._collection.get(
                    limit=self._batch_size, offset=offset, include=[]
                )
                ids = result.get("ids", [])
                if not ids:
                    break
                total += len(ids)
                offset += self._batch_size
            return total
        except Exception as e:
            logger.error(f"[ChromaManager] 统计记录数失败: {e}")
            return 0

    def get_existing_ids(self) -> set:
        """
        分批获取集合中的所有 ID

        Returns:
            已存在的 ID 集合
        """
        existing_ids = set()
        offset = 0
        while True:
            batch_result = self._collection.get(
                limit=self._batch_size, offset=offset, include=[]
            )
            batch_ids = batch_result.get("ids", [])
            if not batch_ids:
                break
            existing_ids.update(batch_ids)
            offset += self._batch_size
        return existing_ids

    def add_texts_in_batches(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
        batch_size: Optional[int] = None,
        desc: str = "写入向量库",
    ) -> int:
        """
        分批向向量库写入文本数据

        Args:
            texts: 文本列表
            metadatas: 对应的元数据列表
            ids: 对应的 ID 列表
            batch_size: 批次大小，默认使用实例配置
            desc: 进度描述（用于日志）

        Returns:
            成功写入的条数
        """
        batch_size = batch_size or self._batch_size
        total = len(texts)
        if total == 0:
            logger.warning(f"[ChromaManager] 无数据需要写入 ({desc})")
            return 0

        success_count = 0
        total_batches = (total + batch_size - 1) // batch_size

        for i in range(0, total, batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_metas = metadatas[i : i + batch_size] if metadatas else None
            batch_ids = ids[i : i + batch_size] if ids else None

            try:
                self._collection.add_texts(
                    texts=batch_texts,
                    metadatas=batch_metas,
                    ids=batch_ids,
                )
                success_count += len(batch_texts)
                logger.info(
                    f"[ChromaManager] {desc} "
                    f"第 {i // batch_size + 1}/{total_batches} 批，"
                    f"本批 {len(batch_texts)} 条"
                )
            except Exception as e:
                logger.error(
                    f"[ChromaManager] {desc} "
                    f"第 {i // batch_size + 1}/{total_batches} 批写入失败: {e}",
                    exc_info=True,
                )

        logger.info(f"[ChromaManager] {desc} 完成，成功写入 {success_count}/{total} 条")
        return success_count

    def clear_and_rebuild_collection(
        self,
        collection_name: str,
        persist_directory: str,
        embedding_function,
        space: str = "cosine",
    ):
        """
        清空向量库集合并重建（通过 Chroma API 删除集合，再重新创建）

        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录
            embedding_function: 嵌入函数
            space: 距离度量空间
        """
        import gc
        import shutil
        import os

        from langchain_chroma import Chroma

        logger.info(f"[ChromaManager] 开始清空并重建集合: {collection_name}")

        if os.path.exists(persist_directory):
            try:
                self._collection._client.delete_collection(collection_name)
                logger.info("[ChromaManager] 已通过 Chroma API 删除集合")
            except Exception as e:
                logger.warning(
                    f"[ChromaManager] Chroma API 删除失败: {e}，尝试文件系统方式..."
                )
                del self._collection
                gc.collect()
                shutil.rmtree(persist_directory)
                logger.info("[ChromaManager] 已通过文件系统删除向量库目录")

        os.makedirs(persist_directory, exist_ok=True)
        self._collection = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": space},
        )
        logger.info(f"[ChromaManager] 集合 {collection_name} 已重建")
