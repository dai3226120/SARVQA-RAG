"""
InternVL 模型 API 客户端
使用 factory.py 的 internvl3_5_8b_model 实例进行多模态调用
"""

from model.factory import internvl3_5_8b_model
from .base_model import BaseAPIClient


class InternVLAPIClient(BaseAPIClient):
    """InternVL API 客户端"""

    def __init__(self):
        super().__init__(internvl3_5_8b_model, model_label="InternVL")

    def print_stats(self):
        """打印调用统计信息"""
        super().print_stats(label="InternVL")
