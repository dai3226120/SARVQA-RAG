"""
知识库 RAG 总结检索服务
实现 BaseRetriever，源自原 rag_service.py (RagSummarizeService)
负责：搜索参考资料 → 拼接上下文 → 提交给模型总结回复
"""
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from rag.base.retriever import BaseRetriever
from rag.stores.knowledge_store import KnowledgeStore
from rag.core.config import rag_config
from model.factory import doubao_seed_20_mini_model
from utils.prompt_loader import load_rag_prompts


class KnowledgeRagService(BaseRetriever):
    """
    知识库 RAG 总结检索服务
    搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
    """

    def __init__(self):
        self._store = KnowledgeStore()
        self._retriever = self._store.get_retriever()
        prompt_text = load_rag_prompts()
        self._prompt_template = PromptTemplate.from_template(prompt_text)
        self._model = doubao_seed_20_mini_model
        self._chain = self._prompt_template | self._model | StrOutputParser()

    def retriever_docs(self, query: str) -> list[Document]:
        """执行向量检索，返回 Document 列表"""
        return self._retriever.invoke(query)

    def rag_summarize(self, query: str) -> str:
        """执行 RAG 总结"""
        context_docs = self.retriever_docs(query)
        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += (
                f"【参考资料{counter}】:参考资料：{doc.page_content} "
                f"| 参考元数据：{doc.metadata}\n"
            )

        return self._chain.invoke({"input": query, "context": context})

    def retrieve(self, query: str) -> str:
        """实现 BaseRetriever 接口"""
        return self.rag_summarize(query)

    def retrieve_context(self, query: str) -> str:
        """
        仅执行 RAG 向量检索并格式化为上下文字符串（不做 LLM 总结）
        供 membership_service / slice_service 的阶段 0 复用

        Args:
            query: 查询文本

        Returns:
            格式化的参考资料字符串；异常时返回空字符串
        """
        try:
            context_docs = self.retriever_docs(query)
            parts = []
            for idx, doc in enumerate(context_docs, 1):
                parts.append(f"【参考资料{idx}】:{doc.page_content}")
            from utils.logger_handler import logger
            logger.info(f"【RAG检索】已完成向量检索，共获取{len(context_docs)}条参考资料")
            return "\n".join(parts)
        except Exception as e:
            from utils.logger_handler import logger
            logger.error(f"RAG检索过程发生异常，跳过RAG检索: {str(e)}")
            return ""
