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
