"""
模型基类
封装 ChatOpenAI 模型实例，提供统一的多模态调用、统计和日志接口
"""

import os
import base64
import time

from langchain_core.messages import HumanMessage

from config import prompt_config
from utils.print_utils import safe_print


class BaseAPIClient:
    """API 客户端基类，通过工厂模型实例调用多模态接口"""

    def __init__(self, model_instance, model_label="model"):
        """
        参数:
            model_instance: factory.py 生成的 ChatOpenAI 模型实例
            model_label: 模型显示标签（用于统计打印）
        """
        self.model = model_instance
        self.model_name = model_instance.model_name
        self.model_label = model_label
        self.call_count = 0
        self.success_count = 0
        self.total_latency = 0.0

    # ==================== 静态工具方法 ====================

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """读取并 base64 编码图像"""
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @staticmethod
    def _build_formatted_prompt(question: str, prompt_template: str = None) -> str:
        """构建格式化后的提示词"""
        template = prompt_template or prompt_config.DEFAULT_PROMPT
        return template.format(question=question)

    # ==================== 核心调用 ====================

    def call(self, image_path: str, question: str, prompt_template: str = None) -> str:
        """
        调用多模态模型获取预测答案

        参数:
            image_path: 图像文件路径
            question: 问题文本
            prompt_template: 提示词模板（可选，默认为默认模板）

        返回:
            str: 预测答案
        """
        call_start_time = time.time()
        self.call_count += 1
        image_name = os.path.basename(image_path)

        if not os.path.exists(image_path):
            safe_print(f"[WARN] 图像文件不存在: {image_name}")
            return "Image file not found"

        try:
            base64_image = self._encode_image(image_path)
            formatted_prompt = self._build_formatted_prompt(question, prompt_template)

            message = HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": formatted_prompt},
            ])

            safe_print(f"[CALL] 调用模型 [{self.call_count}] | 模型: {self.model_name} | 图像: {image_name}")

            response = self.model.invoke([message])

            call_latency = time.time() - call_start_time
            self.total_latency += call_latency
            self.success_count += 1

            safe_print(f"[OK] 调用成功 [{self.call_count}] | 耗时: {call_latency:.2f}秒")
            return response.content

        except Exception as e:
            call_latency = time.time() - call_start_time
            self.total_latency += call_latency
            safe_print(f"[ERROR] 调用失败 [{self.call_count}] | 图像: {image_name} | 耗时: {call_latency:.2f}秒 | 错误: {e}")
            return f"Error: {str(e)}"

    # ==================== 统计接口 ====================

    def get_stats(self) -> dict:
        """获取调用统计信息"""
        avg_latency = self.total_latency / self.call_count if self.call_count > 0 else 0.0
        success_rate = self.success_count / self.call_count * 100 if self.call_count > 0 else 0.0
        return {
            "call_count": self.call_count,
            "success_count": self.success_count,
            "failed_count": self.call_count - self.success_count,
            "success_rate": success_rate,
            "total_latency": self.total_latency,
            "avg_latency": avg_latency
        }

    def print_stats(self, label: str = None):
        """打印调用统计信息"""
        label = label or self.model_label
        stats = self.get_stats()
        safe_print("")
        safe_print("=" * 60)
        safe_print(f"API 调用统计 - {label}")
        safe_print("=" * 60)
        safe_print(f"   - 模型: {self.model_name}")
        safe_print(f"   - 总调用次数: {stats['call_count']}")
        safe_print(f"   - 成功次数: {stats['success_count']}")
        safe_print(f"   - 失败次数: {stats['failed_count']}")
        safe_print(f"   - 成功率: {stats['success_rate']:.2f}%")
        safe_print(f"   - 总耗时: {self.total_latency:.2f}秒")
        safe_print(f"   - 平均耗时: {stats['avg_latency']:.2f}秒/次")
        safe_print("=" * 60)
