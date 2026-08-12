"""
知识库 RAG 总结检索服务
实现 BaseRetriever，源自原 rag_service.py (RagSummarizeService)
负责：搜索参考资料 → 拼接上下文 → 提交给模型总结回复
"""
import re

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from rag.base.retriever import BaseRetriever
from rag.stores.knowledge_store import KnowledgeStore
from rag.core.config import rag_config
from model.factory import doubao_seed_20_mini_model
from utils.prompt_loader import load_rag_prompts
from utils.logger_handler import logger


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

    def retrieve_context(self, query: str, k: int = None, relevance_threshold: float = None) -> str:
        """
        仅执行 RAG 向量检索并格式化为上下文字符串（不做 LLM 总结）
        供 membership_service / slice_service 的阶段 0 复用

        检索质量控制：
        - 多取候选（k×3）后按相关性阈值过滤，剔除低相关噪音片段（如无关章节/表格碎片）；
        - 相同内容去重，避免重复参考资料占满上下文；
        - 清理 txt 分页标记（===== PAGE N =====）噪音。

        Args:
            query: 查询文本
            k: 返回条数（None 用 rag_config.knowledge_retrieve_k）
            relevance_threshold: 相关性阈值（None 用 rag_config.knowledge_relevance_threshold；
                低于该分数的检索结果不注入上下文，防止无关知识误导模型）

        Returns:
            格式化的参考资料字符串；异常时返回空字符串
        """
        try:
            k = k if k is not None else rag_config.knowledge_retrieve_k
            threshold = (relevance_threshold if relevance_threshold is not None
                         else rag_config.knowledge_relevance_threshold)

            # 候选多取（k×3），过滤后仍保证够数
            docs_with_scores = self._store.similarity_search_with_scores(query, k=k * 3)
            seen = set()
            parts = []
            for doc, score in docs_with_scores:
                if score < threshold:
                    continue
                content = doc.page_content.strip()
                if not content or content in seen:
                    continue
                seen.add(content)
                # 清理 txt 分页标记噪音（===== PAGE N =====）
                content = re.sub(r"={5,}\s*PAGE\s*\d+\s*={5,}", "", content).strip()
                parts.append(f"【参考资料{len(parts) + 1}】:{content}")
                if len(parts) >= k:
                    break

            logger.info(f"【RAG检索】已完成向量检索，候选 {len(docs_with_scores)} 条，"
                        f"阈值 {threshold:.2f} 过滤后 {len(parts)} 条")
            return "\n".join(parts)
        except Exception as e:
            logger.error(f"RAG检索过程发生异常，跳过RAG检索: {str(e)}")
            return ""
