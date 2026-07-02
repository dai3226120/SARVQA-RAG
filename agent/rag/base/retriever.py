"""
检索服务统一抽象基类
定义所有检索服务（知识 RAG / 切片检索 / 隶属度混合检索）的通用接口
"""
from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    """
    检索服务抽象基类
    所有 RAG 检索入口统一为 retrieve(query) -> str
    """

    @abstractmethod
    def retrieve(self, query: str) -> str:
        """
        执行检索并返回格式化的检索结果字符串

        Args:
            query: 用户查询文本

        Returns:
            格式化的检索结果（供 LLM 直接使用）
        """
        ...
