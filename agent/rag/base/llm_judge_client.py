"""
LLM 语义判断抽象基类
定义语义匹配判断的统一接口
具体实现通过 model/factory.py 的 LangChain 模型调用
"""
from abc import ABC, abstractmethod
from typing import Optional


class LLMJudgeClient(ABC):
    """
    LLM 语义判断客户端抽象基类
    用于判断预测答案与标准答案是否语义匹配
    """

    @abstractmethod
    def judge(self, question: str, ground_truth: str, predicted: str) -> str:
        """
        判断预测答案是否与标准答案语义匹配

        Args:
            question: 问题文本
            ground_truth: 标准答案
            predicted: 预测答案

        Returns:
            '1'（匹配）或 '0'（不匹配）
        """
        ...
