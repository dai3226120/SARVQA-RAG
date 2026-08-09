"""
全量精确最近邻检索器
以 faiss IndexFlatIP（暴力精确索引）替代 Chroma HNSW 近似检索，返回精确 top-k（余弦相似度）。

- 向量来源：Chroma 集合现有向量（分页导出，预归一化；query 亦归一化，内积=余弦）
- 检索内核：faiss IndexFlatIP —— 全量暴力扫描 + BLAS 加速，与 numpy 矩阵乘数学等价，
  零近似、零漏检；百万级规模下 faiss 的多线程/内存管理优于裸 numpy
- 版本校验：sqlite 的 max_seq_id + count 组合，检测到集合写入变化自动重载
  （覆盖同进程内的新增 / upsert 删旧增新 / 全量重建；跨进程重建由进程重启覆盖）
- 懒加载：首次查询时才导出全量向量并建索引，避免拖慢进程启动
"""
import os
import sqlite3
import threading

import faiss
import numpy as np
from scipy import sparse

from rag.core.config import rag_config
from utils.logger_handler import logger

# faiss 默认按逻辑线程全开（如 32），小规模暴力扫描下 OpenMP 线程同步开销
# 反超并行收益（实测 36k 条: 默认32线程 avg14ms/p95 44ms vs 8线程 avg3.7ms/p95 4.4ms）
faiss.omp_set_num_threads(min(8, os.cpu_count() or 8))


class ExactVectorIndex:
    """基于 faiss IndexFlatIP 的全量精确 top-k 检索器（替代 Chroma HNSW 近似）"""

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
        self._persist_directory = persist_directory
        self._db_path = os.path.join(persist_directory, "chroma.sqlite3")
        self._collection_name = collection_name
        self._embedding_fn = embedding_fn

        self._index: faiss.IndexFlatIP | None = None   # 精确暴力索引（归一化向量，内积=余弦）
        self._ids: np.ndarray | None = None            # 对应日志/切片 id 数组 (N,)
        self._vectors: np.ndarray | None = None        # 归一化向量矩阵 (N, d)，与 _ids 同序（get_all 使用）
        self._metadatas: list | None = None            # 全量 metadatas，与 _ids 同序（get_all 使用）
        self._centers: np.ndarray | None = None        # 类别中心矩阵 (K, d)（get_cluster_centers 缓存）
        self._centers_uniq: np.ndarray | None = None   # 类别 id 列表 (K,)
        self._cluster_ids: np.ndarray | None = None    # 每切片类别 id (N,)（get_cluster_ids 惰性缓存）
        self._centers_version: tuple | None = None     # 类别中心对应的版本（None=未计算；版本变化自动重算）
        self._centers_coll = None                      # 类别中心持久化集合（惰性连接）
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
        """分页导出集合全部向量与元数据，归一化后构建 faiss 精确索引"""
        total = self._collection.count()
        vecs, ids, metas = [], [], []
        offset = 0
        while offset < total:
            r = self._collection.get(
                offset=offset, limit=500, include=["embeddings", "metadatas"]
            )
            vecs.extend(r["embeddings"])
            ids.extend(r["ids"])
            metas.extend(r["metadatas"])
            offset += 500
        if not vecs:
            logger.warning("[ExactVectorIndex] 集合 %s 无向量", self._collection_name)
            self._index = faiss.IndexFlatIP(1)
            self._ids = np.array([], dtype=object)
            self._vectors = np.zeros((0, 1), dtype=np.float32)
            self._metadatas = []
            return
        V = np.ascontiguousarray(np.array(vecs, dtype=np.float32))
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # 零向量兜底，避免除零
        V = V / norms

        self._index = faiss.IndexFlatIP(V.shape[1])
        self._index.add(V)
        self._ids = np.array(ids, dtype=object)
        self._vectors = V
        self._metadatas = metas
        self._cluster_ids = None  # 集合变化后失效，get_cluster_ids 惰性重算
        logger.info(
            "[ExactVectorIndex] %s 全量精确索引已构建: %d 条 (%s)",
            self._collection_name, len(self._ids), V.shape,
        )

    def _ensure_loaded(self):
        """版本变化时自动重载（线程安全）"""
        version = self._version()
        if self._index is not None and version == self._loaded_version:
            return
        with self._lock:
            version = self._version()  # 锁内二次校验，避免并发重复重载
            if self._index is not None and version == self._loaded_version:
                return
            self._load()
            self._loaded_version = version

    # ── 检索 ──

    def count(self) -> int:
        self._ensure_loaded()
        return len(self._ids)

    def get_all(self) -> tuple:
        """全量导出（版本缓存，集合变化自动重载）

        Returns:
            (V, ids, metadatas)：V 为归一化向量矩阵 (N, d) float32，
            ids 为向量库 id 数组 (N,)，metadatas 与 ids 同序
            —— 均为内部只读引用，调用方不得修改
        """
        self._ensure_loaded()
        return self._vectors, self._ids, self._metadatas

    def get_cluster_centers(self) -> tuple:
        """类别中心（持久化优先，版本绑定缓存）

        读取顺序：内存缓存 → 持久化集合（校验 source_version 与当前切片集合版本
        一致则直接用，进程重启不重复计算）→ 重算并回写持久化（旧库无持久化时自动迁移）。
        切片库重建（rebuild_vector_stores.py，md5 检测）后由 refresh_cluster_centers 重算写回

        Returns:
            (centers, uniq)：centers 为归一化类别中心矩阵 (K, d) float32，
            uniq 为类别 id 数组 (K,) int64；无有效类别时返回 (None, None)
        """
        self._ensure_loaded()
        with self._lock:
            if self._centers_version is not None and self._centers_version == self._loaded_version:
                return self._centers, self._centers_uniq
            # 尝试读持久化（校验 source_version 与当前切片集合版本一致）
            persisted = self._load_persisted_cluster_centers()
            if persisted is not None:
                centers, uniq, source_version = persisted
                if source_version == f"{self._loaded_version[0]}:{self._loaded_version[1]}":
                    self._centers, self._centers_uniq = centers, uniq
                    self._centers_version = self._loaded_version
                    return self._centers, self._centers_uniq
            # 重算并回写持久化
            self._centers, self._centers_uniq, self._cluster_ids = self._compute_centers()
            self._centers_version = self._loaded_version
            self._persist_cluster_centers(
                self._centers, self._centers_uniq,
                source_version=f"{self._loaded_version[0]}:{self._loaded_version[1]}",
            )
            return self._centers, self._centers_uniq

    def get_cluster_ids(self) -> np.ndarray:
        """每个切片所属类别 id 数组（与 get_all 同序；无 cluster_id 为 -1；惰性计算并缓存）"""
        self._ensure_loaded()
        with self._lock:
            if self._cluster_ids is None and self._metadatas is not None:
                self._cluster_ids = np.array(
                    [
                        m.get("cluster_id", -1) if isinstance(m, dict) else -1
                        for m in self._metadatas
                    ],
                    dtype=np.int64,
                )
        return self._cluster_ids

    def refresh_cluster_centers(self, source_md5=None) -> tuple:
        """强制重算类别中心并写回持久化集合（切片库重建后调用）

        Args:
            source_md5: 切片库源文件（sar_slices.csv）的 md5，随中心持久化，
                供 rebuild 脚本检测切片库是否变化

        Returns:
            (centers, uniq)：归一化类别中心矩阵与类别 id 列表
        """
        self._ensure_loaded()
        with self._lock:
            self._centers, self._centers_uniq, self._cluster_ids = self._compute_centers()
            self._centers_version = self._loaded_version
            self._persist_cluster_centers(
                self._centers, self._centers_uniq,
                source_version=f"{self._loaded_version[0]}:{self._loaded_version[1]}",
                source_md5=source_md5,
            )
            return self._centers, self._centers_uniq

    def get_persisted_source_md5(self) -> str | None:
        """读取持久化类别中心对应的切片库源文件 md5（轻量，不加载切片索引）

        供 rebuild_vector_stores.py 判断切片库文件是否变化（变化才重建）
        """
        try:
            data = self._centers_collection().get(include=["metadatas"], limit=1)
            if data["ids"] and data["metadatas"]:
                return data["metadatas"][0].get("source_md5") or None
        except Exception:
            pass
        return None

    def _centers_collection(self):
        """类别中心持久化集合（chromadb 原生，仅存中心向量+元数据；惰性连接）"""
        if self._centers_coll is None:
            import chromadb
            client = chromadb.PersistentClient(path=self._persist_directory)
            self._centers_coll = client.get_or_create_collection(
                name=rag_config.cluster_centers_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._centers_coll

    def _load_persisted_cluster_centers(self) -> tuple | None:
        """从持久化集合读取类别中心

        Returns:
            (centers, uniq, source_version)：归一化中心矩阵、类别 id 列表、来源版本；
            集合不存在/为空/读取失败时返回 None
        """
        try:
            data = self._centers_collection().get(include=["embeddings", "metadatas"])
            if not data["ids"]:
                return None
            metas, emb = data["metadatas"], data["embeddings"]
            order = sorted(range(len(metas)), key=lambda i: int(metas[i]["cluster_id"]))
            centers = np.array([emb[i] for i in order], dtype=np.float32)
            norms = np.linalg.norm(centers, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            centers = centers / norms
            uniq = np.array([int(metas[i]["cluster_id"]) for i in order], dtype=np.int64)
            return centers, uniq, metas[order[0]].get("source_version")
        except Exception as e:
            logger.warning("[ExactVectorIndex] 持久化类别中心读取失败: %s", e)
            return None

    def _persist_cluster_centers(self, centers, uniq, source_version, source_md5=None):
        """覆盖写回类别中心到持久化集合（重建/重算后调用）"""
        try:
            coll = self._centers_collection()
            # 清空旧记录（chromadb 1.x delete 需显式 ids，不支持 where=None）
            old = coll.get(include=["metadatas"])
            if old["ids"]:
                coll.delete(ids=old["ids"])
            if centers is None or len(uniq) == 0:
                return
            coll.add(
                ids=[str(int(c)) for c in uniq],
                embeddings=[c.tolist() for c in centers],
                metadatas=[
                    {
                        "cluster_id": int(c),
                        "source_version": str(source_version),
                        "source_md5": source_md5 or "",
                    }
                    for c in uniq
                ],
            )
            logger.info("[ExactVectorIndex] 类别中心已持久化: %d 个类别", len(uniq))
        except Exception as e:
            logger.warning("[ExactVectorIndex] 类别中心持久化失败: %s", e)

    def _compute_centers(self) -> tuple:
        """按 metadatas 的 cluster_id 分组求类别中心（平均向量 → 归一化）

        Returns:
            (centers, uniq, cluster_ids)：centers 归一化中心矩阵 (K, d)，
            uniq 类别 id 数组 (K,)，cluster_ids 每切片类别 id 数组 (N,)；无有效类别时 (None, None, None)
        """
        if self._metadatas is None or len(self._ids) == 0:
            return None, None, None
        cluster_ids = np.array(
            [
                m.get("cluster_id", -1) if isinstance(m, dict) else -1
                for m in self._metadatas
            ],
            dtype=np.int64,
        )
        valid = cluster_ids >= 0
        if not valid.any():
            return None, None, cluster_ids
        v = self._vectors if valid.all() else self._vectors[valid]
        uniq = np.unique(cluster_ids[valid])
        pos = np.searchsorted(uniq, cluster_ids[valid])
        # 稀疏 one-hot 矩阵乘求组内向量和：只算非零位置，远快于 np.add.at / np.add.reduceat
        # （实测 36k×1024、K=500：np.add.at 1377ms / reduceat 425ms / 稀疏矩阵乘 9ms）
        one_hot = sparse.csr_matrix(
            (np.ones(len(v), dtype=np.float32), (pos, np.arange(len(v)))),
            shape=(len(uniq), len(v)),
        )
        sums = np.asarray(one_hot @ v)  # (K, d) 每类向量和
        counts = np.bincount(pos).astype(np.float64)
        centers = sums / counts[:, None]
        norms = np.linalg.norm(centers, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # 零向量兜底，避免除零
        return centers / norms, uniq, cluster_ids

    def search(self, query_embedding, k: int) -> list[tuple[str, float]]:
        """全量精确余弦 top-k。返回 [(id, score)] 按相似度降序，score ∈ [-1, 1]"""
        self._ensure_loaded()
        qe = np.ascontiguousarray(np.array(query_embedding, dtype=np.float32).reshape(1, -1))
        norm = np.linalg.norm(qe)
        if norm > 0:
            qe = qe / norm
        scores, idx = self._index.search(qe, k)
        return [
            (str(self._ids[i]), float(scores[0][j]))
            for j, i in enumerate(idx[0])
            if i != -1  # faiss 不足 k 条时以 -1 填充
        ]

    def search_text(self, query: str, k: int, qe=None) -> list[tuple[dict, float]]:
        """嵌入 query 后全量精确检索，并取回 metadata。返回 [(metadata, score)] 降序

        qe: 可选预嵌入向量（复用上层已计算的 query embedding，省一次嵌入排队）
        """
        if self._embedding_fn is None:
            raise RuntimeError("[ExactVectorIndex] 未配置 embedding_fn，无法使用 search_text")
        if qe is None:
            qe = self._embedding_fn.embed_query(query)
        hits = self.search(qe, k)
        if not hits:
            return []
        ids = [h[0] for h in hits]
        metas = self._collection.get(ids=ids, include=["metadatas"])["metadatas"]
        return list(zip(metas, [h[1] for h in hits]))
