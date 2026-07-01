"""
models 模块
统一导出所有 API 客户端及便捷函数
"""

from .doubao_seed import DoubaoAPIClient
from .internvl import InternVLAPIClient
from .agent_doubao_seed import AgentDoubaoAPIClient
from .agent_internvl import AgentInternVLAPIClient

# 全局实例（保持兼容原 api_client.py 接口）
doubao_client = DoubaoAPIClient()
internvl_client = InternVLAPIClient()
agent_doubao_client = AgentDoubaoAPIClient()
agent_internvl_client = AgentInternVLAPIClient()


# ==================== 便捷函数 ====================

def call_doubao_api(image_path, question, prompt_template=None):
    """便捷函数：调用豆包 API"""
    return doubao_client.call(image_path, question, prompt_template)


def call_internvl_api(image_path, question, prompt_template=None):
    """便捷函数：调用 InternVL API"""
    return internvl_client.call(image_path, question, prompt_template)


def call_agent_doubao_api(image_path, question, prompt_template=None):
    """便捷函数：调用 Agent 豆包 API"""
    return agent_doubao_client.call(image_path, question, prompt_template)


def call_agent_internvl_api(image_path, question, prompt_template=None):
    """便捷函数：调用 Agent InternVL API"""
    return agent_internvl_client.call(image_path, question, prompt_template)
