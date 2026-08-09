"""
SAR 切片检索服务
实现 BaseRetriever，源自原 rag_rscsv_service_rscsv.py (RscsvServiceRscsv)
负责：基础切片向量检索（全量精确检索 + 相似度得分归一化）
"""
import os
import threading
import time

from rag.base.retriever import BaseRetriever
from rag.stores.slice_store import SliceStore
from rag.builders.slice_builder import SliceBuilder
from rag.services.knowledge_service import KnowledgeRagService
from rag.core.config import rag_config
from rag.core.exact_index import ExactVectorIndex
from model.factory import huggingface_embed_model
from utils.logger_handler import logger


def retrieve_basic_slices(slice_index, store, query: str, slice_k: int, qe=None) -> dict:
    """
    基础切片检索统一内核（全量精确检索算法唯一实现点）
    供 SliceRetrievalService 阶段1 与 MembershipHybridService 阶段2（降级）复用；
    后续改造检索算法只需修改本函数，两个阶段同步生效

    Args:
        slice_index: ExactVectorIndex 全量精确检索器
        store: SliceStore（按 id 精确取文档内容）
        query: 查询文本
        slice_k: 检索返回的切片数量（检索多少就送多少，全部保留）
        qe: 预嵌入的 query 向量（复用上游结果时传入，None 时内部嵌入）

    Returns:
        dict: {
            "result_str": 格式化切片检索结果字符串（不含 SLICE_IDS 标记与 RAG 前缀）
            "slices_trace": [{slice_id, score(归一化), score_type, content}]，按归一化分数降序
            "hit": 是否命中
            "timing": {"embed_latency", "search_latency", "get_latency", "total_latency"}（ms）
        }
    """
    # 嵌入 → 全量精确检索（faiss IndexFlatIP，非 HNSW 近似）
    _t0 = time.perf_counter()
    if qe is None:
        qe = huggingface_embed_model.embed_query(query)
    _t1 = time.perf_counter()
    slice_results = slice_index.search(qe, slice_k)  # [(slice_id, score)] 降序
    _t2 = time.perf_counter()

    if not slice_results:
        return {
            "result_str": "未检索到相关遥感问答参考资料。",
            "slices_trace": [],
            "hit": False,
            "timing": {
                "total_latency": (time.perf_counter() - _t0) * 1000,
                "embed_latency": (_t1 - _t0) * 1000,
                "search_latency": (_t2 - _t1) * 1000,
                "get_latency": 0.0,
            },
        }

    # 按 id 精确取文档内容（与 hits 顺序一致）；检索多少就送多少，全部保留
    top_ids = [h[0] for h in slice_results]
    got = store.get_by_ids(top_ids)
    _t3 = time.perf_counter()
    documents = got.get("documents") or []
    metadatas = got.get("metadatas") or []

    # 归一化相似度分数到 [0, 1]（min-max，保持原始降序）
    raw_scores = [h[1] for h in slice_results]
    min_score = min(raw_scores) if raw_scores else -1
    max_score = max(raw_scores) if raw_scores else 1

    def normalize_score(score: float) -> float:
        if max_score == min_score:
            return 0.5 if max_score > 0 else 0.0
        normalized = (score - min_score) / (max_score - min_score)
        return max(0.0, min(1.0, normalized))

    sorted_results = sorted(
        [
            (i, normalize_score(score), score)
            for i, score in enumerate(raw_scores)
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    # 切片记录（归一化得分），顺序与显示文本一致
    slices_trace = [
        {
            "slice_id": (
                metadatas[i].get("slice_id", top_ids[i])
                if isinstance(metadatas[i], dict)
                else top_ids[i]
            ),
            "score": norm_score,
            "score_type": "similarity",
            "content": documents[i],
        }
        for i, norm_score, _ in sorted_results
        if i < len(documents)
    ]

    content = "\n---\n".join(
        [
            f"相似度得分: {norm_score:.4f}\n{documents[i]}"
            for i, norm_score, _ in sorted_results
            if i < len(documents)
        ]
    )
    return {
        "result_str": f"【匹配基础切片】(全量精确检索{slice_k}条)  \n{content}",
        "slices_trace": slices_trace,
        "hit": True,
        "timing": {
            "total_latency": (_t3 - _t0) * 1000,
            "embed_latency": (_t1 - _t0) * 1000,
            "search_latency": (_t2 - _t1) * 1000,
            "get_latency": (_t3 - _t2) * 1000,
        },
    }


class SliceRetrievalService(BaseRetriever):
    """
    SAR 切片检索服务（仅基础切片检索，无隶属度）
    包含相似度得分归一化和线程局部切片 ID 记录
    """

    # 线程局部存储：确保多线程环境下数据隔离
    # agent 内部工具调用可能在子线程中，需通过返回值中的标记传递切片 ID
    _thread_local = threading.local()

    def __init__(self):
        self._builder = SliceBuilder()
        self._store = self._builder.store
        self._knowledge_service = KnowledgeRagService()
        self._slice_k = rag_config.slice_k
        self._enable_rag_context = rag_config.enable_rag_context
        # 切片库全量精确检索器（faiss 暴力 top-k，替代 Chroma HNSW 近似）
        self._slice_index = ExactVectorIndex(
            collection=self._store.collection._collection,
            persist_directory=rag_config.persist_directory,
            collection_name=rag_config.slices_collection_name,
            embedding_fn=huggingface_embed_model,
        )
        self._last_trace: dict | None = None

    def get_last_trace(self) -> dict | None:
        """获取最近一次检索的过程记录（耗时拆分，供 UI/统计展示）"""
        return self._last_trace

    def hybrid_retrieve(self, query: str, slice_k: int = None) -> str:
        """
        执行基础切片检索（带相似度得分归一化）

        Args:
            query: 查询文本
            slice_k: 检索返回的切片数量（检索多少就送多少，全部保留）

        Returns:
            格式化的检索结果字符串（含切片 ID 标记）
        """
        slice_k = slice_k or self._slice_k

        # ==========================================
        # 阶段 0: RAG 向量检索（封装在 _retrieve_rag_context）
        # ==========================================
        rag_context = self._retrieve_rag_context(query)

        # ==========================================
        # 阶段 1: 基础切片检索（封装在 _retrieve_by_similarity）
        # ==========================================
        result_str = self._retrieve_by_similarity(query, slice_k)

        if rag_context:
            return f"【RAG检索参考】\n{rag_context}\n\n==============================\n\n{result_str}"
        return result_str

    def _retrieve_rag_context(self, query: str) -> str:
        """
        阶段 0: RAG 向量检索（与 membership_service._retrieve_rag_context 实现一致）
        通过 chroma.yml → retrieval.enable_rag_context 控制开关；
        封装知识库服务的上下文检索（检索 → 格式化为参考资料字符串）
        """
        return self._knowledge_service.retrieve_context(query) if self._enable_rag_context else ""

    def _retrieve_by_similarity(self, query: str, slice_k: int) -> str:
        """
        阶段 1: 基础切片检索（统一内核 retrieve_basic_slices + 本服务记录外壳）

        Args:
            query: 查询文本
            slice_k: 检索返回的切片数量（检索多少就送多少，全部保留）

        Returns:
            格式化的切片检索结果字符串（含 SLICE_IDS 标记）
        """
        data = retrieve_basic_slices(self._slice_index, self._store, query, slice_k)
        timing = data["timing"]

        if not data["hit"]:
            SliceRetrievalService._thread_local.slice_ids = []
            self._last_trace = {
                "total_latency": timing["total_latency"],
                "stage1_latency": timing["total_latency"],  # 对齐 RetrievalTrace 字段（无阶段0/2）
                "embed_latency": timing["embed_latency"],
                "search_latency": timing["search_latency"],
                "get_latency": timing["get_latency"],
                "decision": "slice_miss",
            }
            return "未检索到相关遥感问答参考资料。\n<!-- SLICE_IDS: -->"

        # 记录检索使用的切片 ID（工具回传标记）
        slice_ids = [item["slice_id"] for item in data["slices_trace"]]
        SliceRetrievalService._thread_local.slice_ids = slice_ids
        self._last_trace = {
            "total_latency": timing["total_latency"],
            "stage1_latency": timing["total_latency"],
            "embed_latency": timing["embed_latency"],
            "search_latency": timing["search_latency"],
            "get_latency": timing["get_latency"],
            "decision": "slice_hit",
        }
        return f"{data['result_str']}\n<!-- SLICE_IDS: {','.join(slice_ids)} -->"

    def retrieve(self, query: str) -> str:
        """实现 BaseRetriever 接口"""
        return self.hybrid_retrieve(query)

    @classmethod
    def get_last_retrieved_slice_ids(cls) -> list:
        """获取当前线程最后检索的切片 ID 列表"""
        if not hasattr(cls._thread_local, "slice_ids"):
            cls._thread_local.slice_ids = []
        return cls._thread_local.slice_ids.copy()

    @classmethod
    def reset_last_retrieved_slice_ids(cls):
        """重置当前线程最后检索的切片 ID 列表"""
        cls._thread_local.slice_ids = []
