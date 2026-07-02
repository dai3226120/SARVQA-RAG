"""
真实 RAG 引擎包装器
源自原 rag_rsfit_builder.py RealRAGEngine
用于 benchmark 评估场景，提供 search() 接口
"""
import os

from rag.stores.slice_store import SliceStore
from rag.builders.slice_builder import SliceBuilder
from rag.core.config import rag_config
from utils.logger_handler import logger


class RealRAGEngine:
    """真实的 RAG 引擎包装器，使用 SliceStore 的切片向量库"""

    def __init__(self):
        self._builder = SliceBuilder()
        self._store = self._builder.store
        self._slice_k = rag_config.slice_k
        self._top_p = rag_config.top_p

    def search(self, query, k=None):
        """
        从向量库中检索相关切片

        Args:
            query: 查询问题
            k: 返回结果数量（已废弃，使用配置中的 slice_k）

        Returns:
            list: 检索到的切片列表（按相似度降序排序后取前 top_p 条）
        """
        slice_results = self._store.similarity_search_with_scores(
            query, k=self._slice_k
        )

        if not slice_results:
            return []

        sorted_results = sorted(slice_results, key=lambda x: x[1], reverse=True)
        top_results = sorted_results[: self._top_p]

        results = []
        for doc, score in top_results:
            slice_id = getattr(doc, "id", None) or doc.metadata.get("slice_id")

            if not slice_id:
                slice_id = str(hash(doc.page_content))

            content = doc.page_content
            results.append({
                "id": slice_id,
                "content": content,
                "similarity_score": float(score),
            })
        return results
