"""
SAR 切片检索服务
实现 BaseRetriever，源自原 rag_rscsv_service_rscsv.py (RscsvServiceRscsv)
负责：基础切片向量检索（带相似度得分归一化）
"""
import os
import threading

from rag.base.retriever import BaseRetriever
from rag.stores.slice_store import SliceStore
from rag.builders.slice_builder import SliceBuilder
from rag.services.knowledge_service import KnowledgeRagService
from rag.core.config import rag_config
from utils.logger_handler import logger


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
        self._top_p = rag_config.top_p
        self._enable_rag_context = rag_config.enable_rag_context

    def hybrid_retrieve(self, query: str, slice_k: int = None, top_p: int = None) -> str:
        """
        执行基础切片检索（带相似度得分归一化）

        Args:
            query: 查询文本
            slice_k: 检索返回的切片数量
            top_p: 最终保留的结果数量

        Returns:
            格式化的检索结果字符串（含切片 ID 标记）
        """
        slice_k = slice_k or self._slice_k
        top_p = top_p or self._top_p

        # ==========================================
        # 阶段 0: RAG 向量检索（通过 chroma.yml → retrieval.enable_rag_context 控制）
        # ==========================================
        rag_context = self._knowledge_service.retrieve_context(query) if self._enable_rag_context else ""

        slice_results = self._store.similarity_search_with_scores(query, k=slice_k)
        if slice_results:
            # 归一化相似度分数到 [0, 1]
            raw_scores = [score for _, score in slice_results]
            min_score = min(raw_scores) if raw_scores else -1
            max_score = max(raw_scores) if raw_scores else 1

            def normalize_score(score: float) -> float:
                if max_score == min_score:
                    return 0.5 if max_score > 0 else 0.0
                normalized = (score - min_score) / (max_score - min_score)
                return max(0.0, min(1.0, normalized))

            # 按归一化分数排序
            normalized_results = [
                (doc, normalize_score(score), score) for doc, score in slice_results
            ]
            sorted_results = sorted(normalized_results, key=lambda x: x[1], reverse=True)
            top_results = sorted_results[:top_p]

            # 记录检索使用的切片 ID
            SliceRetrievalService._thread_local.slice_ids = [
                doc.metadata.get("slice_id", str(i))
                for i, (doc, _, _) in enumerate(top_results)
            ]

            content = "\n---\n".join(
                [
                    f"相似度得分: {norm_score:.4f}\n{doc.page_content}"
                    for doc, norm_score, _ in top_results
                ]
            )
            slice_ids_str = ",".join(
                [
                    doc.metadata.get("slice_id", str(i))
                    for i, (doc, _, _) in enumerate(top_results)
                ]
            )
            return (
                f"【RAG检索参考】\n{rag_context}\n\n==============================\n\n"
                f"【匹配基础切片】(共检索{slice_k}条，"
                f"按相似度排序后保留{len(top_results)}条)  \n{content}\n"
                f"<!-- SLICE_IDS: {slice_ids_str} -->"
            ) if rag_context else (
                f"【匹配基础切片】(共检索{slice_k}条，"
                f"按相似度排序后保留{len(top_results)}条)  \n{content}\n"
                f"<!-- SLICE_IDS: {slice_ids_str} -->"
            )

        SliceRetrievalService._thread_local.slice_ids = []
        no_result = "未检索到相关遥感问答参考资料。\n<!-- SLICE_IDS: -->"
        if rag_context:
            return f"【RAG检索参考】\n{rag_context}\n\n==============================\n\n{no_result}"
        return no_result

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
