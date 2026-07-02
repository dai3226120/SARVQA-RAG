"""
向量库统一抽象基类
定义所有向量库（知识库/切片/日志）的通用接口
"""
from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.documents import Document


class BaseVectorStore(ABC):
    """
    向量库抽象基类
    统一 add / get / count / search / rebuild / clear 接口
    具体实现由 KnowledgeStore / SliceStore / LogManager 各自完成
    """

    @abstractmethod
    def get_retriever(self, k: int = 5):
        """获取 LangChain 检索器"""
        ...

    @abstractmethod
    def add_documents(self, documents: list[Document], batch_size: int = 500) -> int:
        """分批写入文档，返回写入条数"""
        ...

    @abstractmethod
    def get_by_ids(self, ids: list[str]) -> dict:
        """
        按 ID 精确获取文档
        Returns: {'documents': [...], 'metadatas': [...]}
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """获取记录总数"""
        ...

    @abstractmethod
    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        """语义相似度检索（不含分数）"""
        ...

    @abstractmethod
    def similarity_search_with_scores(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        """语义相似度检索（含分数）"""
        ...
