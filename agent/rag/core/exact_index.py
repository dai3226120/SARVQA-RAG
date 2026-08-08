"""
全量精确最近邻检索器
以 numpy 矩阵乘替代 Chroma HNSW 近似检索，返回精确 top-k（余弦相似度）。

- 向量来源：Chroma 集合现有向量（分页导出，预归一化；query 亦归一化，余弦=点积）
- 版本校验：sqlite 的 max_seq_id + count 组合，检测到集合写入变化自动重载
  （覆盖同进程内的新增 / upsert 删旧增新 / 全量重建；跨进程重建由进程重启覆盖）
- 懒加载：首次查询时才导出全量向量，避免拖慢进程启动
"""
import os
import sqlite3
import threading

import numpy as np

from utils.logger_handler import logger


class ExactVectorIndex:
    """基于 numpy 的全量精确 top-k 检索器（替代 Chroma HNSW 近似）"""

    def __init__(
        self,
        collection,
        persist_directory: str,
        collection_name: str,
        embedding_fn=None,
    ):
        """
        Args:
            collection: chromadb 底层 Collection（Chroma 实例的 _collection）
            persist_directory: Chroma 持久化目录（含 chroma.sqlite3）
            collection_name: 集合名（用于 sqlite 定位该集合的 segment）
            embedding_fn: query 嵌入函数（embed_query(q) -> list[float]），
                None 时退化：search_text 不可用，只能传显式 embedding 的 search()
        """
        self._collection = collection
        self._db_path = os.path.join(persist_directory, "chroma.sqlite3")
        self._collection_name = collection_name
        self._embedding_fn = embedding_fn

        self._matrix: np.ndarray | None = None   # 归一化向量矩阵 (N, d)
        self._ids: np.ndarray | None = None      # 对应日志/切片 id 数组 (N,)
        self._loaded_version: tuple = (-1, -1)
        self._lock = threading.Lock()

    # ── 版本管理 ──

    def _version(self) -> tuple:
        """(count, max_seq) 版本号：新增/替换/重建任一变化都会改变至少一项"""
        try:
            con = sqlite3.connect(self._db_path)
            try:
                cur = con.cursor()
                cur.execute(
                    """SELECT s.id FROM segments s
                       JOIN collections c ON s.collection = c.id
                       WHERE c.name = ?""",
                    (self._collection_name,),
                )
                segment_ids = [r[0] for r in cur.fetchall()]
                if not segment_ids:
                    return (-1, -1)
                placeholders = ",".join("?" * len(segment_ids))
                cur.execute(
                    f"SELECT COUNT(*), COALESCE(MAX(seq_id), 0) FROM max_seq_id "
                    f"WHERE segment_id IN ({placeholders})",
                    segment_ids,
                )
                row = cur.fetchone()
                return (row[0], row[1])
            finally:
                con.close()
        except Exception as e:
            logger.warning("[ExactVectorIndex] 版本读取失败: %s", e)
            return (-1, -1)

    def _load(self):
        """分页导出集合全部向量并预归一化"""
        total = self._collection.count()
        vecs, ids = [], []
        offset = 0
        while offset < total:
            r = self._collection.get(offset=offset, limit=500, include=["embeddings"])
            vecs.extend(r["embeddings"])
            ids.extend(r["ids"])
            offset += 500
        if not vecs:
            logger.warning("[ExactVectorIndex] 集合 %s 无向量", self._collection_name)
            self._matrix = np.zeros((0, 1), dtype=np.float32)
            self._ids = np.array([], dtype=object)
            return
        V = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # 零向量兜底，避免除零
        self._matrix = V / norms
        self._ids = np.array(ids, dtype=object)
        logger.info(
            "[ExactVectorIndex] %s 全量向量已加载: %d 条 (%s)",
            self._collection_name, len(self._ids), self._matrix.shape,
        )

    def _ensure_loaded(self):
        """版本变化时自动重载（线程安全）"""
        version = self._version()
        if self._matrix is not None and version == self._loaded_version:
            return
        with self._lock:
            version = self._version()  # 锁内二次校验，避免并发重复重载
            if self._matrix is not None and version == self._loaded_version:
                return
            self._load()
            self._loaded_version = version

    # ── 检索 ──

    def count(self) -> int:
        self._ensure_loaded()
        return len(self._ids)

    def search(self, query_embedding, k: int) -> list[tuple[str, float]]:
        """全量精确余弦 top-k。返回 [(id, score)] 按相似度降序，score ∈ [-1, 1]"""
        self._ensure_loaded()
        qe = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(qe)
        if norm > 0:
            qe = qe / norm
        scores = self._matrix @ qe
        top_idx = np.argsort(scores)[-k:][::-1]
        return [
            (str(self._ids[i]), float(scores[i]))
            for i in top_idx
        ]

    def search_text(self, query: str, k: int) -> list[tuple[dict, float]]:
        """嵌入 query 后全量精确检索，并取回 metadata。返回 [(metadata, score)] 降序"""
        if self._embedding_fn is None:
            raise RuntimeError("[ExactVectorIndex] 未配置 embedding_fn，无法使用 search_text")
        qe = self._embedding_fn.embed_query(query)
        hits = self.search(qe, k)
        if not hits:
            return []
        ids = [h[0] for h in hits]
        metas = self._collection.get(ids=ids, include=["metadatas"])["metadatas"]
        return list(zip(metas, [h[1] for h in hits]))
