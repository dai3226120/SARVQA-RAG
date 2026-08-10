"""
models 模块
统一导出所有 API 客户端及便捷函数
"""

from .doubao_seed import DoubaoAPIClient
from .internvl import InternVLAPIClient
from .main_agent import MainAgentClient
from .qwen37_plus import Qwen37PlusAPIClient
from .vllm_models import tinygptv_client, model_2_client, model_3_client  # vLLM 部署模型（tinygptv / 占位）

# 非 Agent 全局实例
doubao_client = DoubaoAPIClient()
internvl_client = InternVLAPIClient()
qwen37_plus_client = Qwen37PlusAPIClient()

# Agent 全局实例
doubao_agent_client = MainAgentClient("mainagent")
internvl_agent_client = MainAgentClient("mainagent_internVL")
doubao_agent_rscsv_client = MainAgentClient("mainagent_rscsv")
internvl_agent_rscsv_client = MainAgentClient("mainagent_internVL_rscsv")
tinygptv_agent_client = MainAgentClient("mainagent_tinygptv")          # TinyGPT-V Agent
tinygptv_agent_rscsv_client = MainAgentClient("mainagent_tinygptv_rscsv")  # TinyGPT-V RSCSV Agent


# ==================== 便捷函数 ====================

def call_doubao_api(image_path, question, prompt_template=None):
    """便捷函数：调用豆包 API"""
    return doubao_client.call(image_path, question, prompt_template)


def call_internvl_api(image_path, question, prompt_template=None):
    """便捷函数：调用 InternVL API"""
    return internvl_client.call(image_path, question, prompt_template)


def call_doubao_agent(image_path, question, prompt_template=None):
    """便捷函数：调用豆包 Agent"""
    return doubao_agent_client.call(image_path, question, prompt_template)


def call_internvl_agent(image_path, question, prompt_template=None):
    """便捷函数：调用 InternVL Agent"""
    return internvl_agent_client.call(image_path, question, prompt_template)


def call_qwen37_plus(image_path, question, prompt_template=None):
    """便捷函数：调用 Qwen3.7-Plus API"""
    return qwen37_plus_client.call(image_path, question, prompt_template)
