"""
Agent 版 InternVL 模型 API 客户端
使用 Agent 专用提示词
"""

from .internvl import InternVLAPIClient
from config import prompt_config


class AgentInternVLAPIClient(InternVLAPIClient):
    """Agent 版 InternVL API 客户端，默认使用 Agent 提示词"""

    def call(self, image_path: str, question: str, prompt_template: str = None) -> str:
        """
        调用 InternVL Agent API 获取预测答案
        默认使用 Agent 提示词（150 词限制）
        """
        template = prompt_template or prompt_config.AGENT_PROMPT
        return super().call(image_path, question, template)

    def print_stats(self):
        """打印调用统计信息"""
        from .base_model import BaseAPIClient
        BaseAPIClient.print_stats(self, label="Agent InternVL")
