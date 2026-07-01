"""
Doubao（豆包）模型 API 客户端
使用 factory.py 的 doubao_seed_20_mini_model 实例进行多模态调用
"""

from model.factory import doubao_seed_20_mini_model
from .base_model import BaseAPIClient


class DoubaoAPIClient(BaseAPIClient):
    """豆包 API 客户端"""

    def __init__(self):
        super().__init__(doubao_seed_20_mini_model, model_label="豆包")

    def print_stats(self):
        """打印调用统计信息"""
        super().print_stats(label="豆包")
