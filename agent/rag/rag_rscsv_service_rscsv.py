import os
import threading
from rag.rag_rscsv_builder import RscsvBuilder
from rag.rag_service import RagSummarizeService
from utils.config_handler import chroma_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


class RscsvServiceRscsv:
    # 使用线程局部存储确保多线程环境下数据隔离
    # 注意：agent内部工具调用可能在子线程中，此时需要通过返回值中的标记传递切片ID
    _thread_local = threading.local()

    def __init__(self):
        # 1. 基础构建器与切片集合
        self.builder = RscsvBuilder()
        self.slice_collection = self.builder.slice_collection
        
        # 2. 初始化 RAG 总结服务
        self.rag_summarize_service = RagSummarizeService()
        
        # 3. 配置参数
        self.top_p = int(chroma_conf.get("top_p", 3))
        self.slice_k = int(chroma_conf.get("slice_k", 10))

    def hybrid_retrieve(self, query: str, k: int = 1) -> str:
        # ==========================================
        # 阶段 0: RAG 向量检索
        # ==========================================
        try:
            context_docs = self.rag_summarize_service.retriever_docs(query)
            rag_context_parts = []
            for idx, doc in enumerate(context_docs, 1):
                rag_context_parts.append(f"【参考资料{idx}】:{doc.page_content}")
            rag_context = "\n".join(rag_context_parts)
            logger.info(f"【RAG检索】已完成向量检索，共获取{len(context_docs)}条参考资料")
        except Exception as e:
            logger.error(f"RAG检索过程发生异常，跳过RAG检索: {str(e)}")
            rag_context = ""

        # ==========================================
        # 阶段 1: 基础切片检索（带相似度得分）
        # ==========================================
        slice_results = self.slice_collection.similarity_search_with_relevance_scores(query, k=k)
        if slice_results:
            # 按相似度得分降序排序
            sorted_results = sorted(slice_results, key=lambda x: x[1], reverse=True)
            
            # 取 top_p 条结果
            top_results = sorted_results[:self.top_p]
            
            # 记录本次检索使用的切片ID（使用线程局部存储）
            RscsvServiceRscsv._thread_local.slice_ids = [
                doc.metadata.get('slice_id', str(i)) for i, (doc, score) in enumerate(top_results)
            ]
            
            # 拼接检索结果（包含相似度得分）
            content = "\n---\n".join(
                [f"相似度得分: {score:.4f}\n{doc.page_content}" for doc, score in top_results]
            )
            slice_ids_str = ",".join([
                doc.metadata.get('slice_id', str(i)) for i, (doc, score) in enumerate(top_results)
            ])
            rscsv_result = f"【匹配基础切片】(共检索{k}条，按相似度排序后保留{len(top_results)}条)  \n{content}\n<!-- SLICE_IDS: {slice_ids_str} -->"
            if rag_context:
                return f"【RAG检索参考】\n{rag_context}\n\n==============================\n\n{rscsv_result}"
            return rscsv_result

        # 使用线程局部存储清空切片ID
        RscsvServiceRscsv._thread_local.slice_ids = []
        
        if rag_context:
            return f"【RAG检索参考】\n{rag_context}\n\n==============================\n\n未检索到相关遥感问答参考资料。\n<!-- SLICE_IDS: -->"
        return "未检索到相关遥感问答参考资料。\n<!-- SLICE_IDS: -->"

    def retrieve(self, query: str) -> str:
        # 确保配置中的 k 为整数
        return self.hybrid_retrieve(query, k=self.slice_k)

    @classmethod
    def get_last_retrieved_slice_ids(cls) -> list:
        """获取当前线程最后检索的切片ID列表"""
        if not hasattr(cls._thread_local, 'slice_ids'):
            cls._thread_local.slice_ids = []
        return cls._thread_local.slice_ids.copy()

    @classmethod
    def reset_last_retrieved_slice_ids(cls):
        """重置当前线程最后检索的切片ID列表"""
        cls._thread_local.slice_ids = []


if __name__ == "__main__":
    service = RscsvServiceRscsv()
    print(service.retrieve("Does the radar image show any weather-related disturbances, like storms or rain?"))