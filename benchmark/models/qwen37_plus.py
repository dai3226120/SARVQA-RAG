"""
Qwen3.7-Plus API 客户端
通过 DashScope MultiModalConversation API 调用多模态模型
"""

import os
import time

import dashscope

from utils.config_handler import model_conf
from utils.print_utils import safe_print


class Qwen37PlusAPIClient:
    """Qwen3.7-Plus 多模态 API 客户端，使用 DashScope MultiModalConversation"""

    def __init__(self):
        self.model_name = model_conf['qwen37_plus_model_name']
        self.model_label = "Qwen3.7-Plus"

        # 设置自定义 MaaS endpoint
        dashscope.base_http_api_url = model_conf['qwen37_plus_api_endpoint']

        self.api_key = os.environ.get('DASHSCOPE_API_KEY', '')
        self.call_count = 0
        self.success_count = 0
        self.total_latency = 0.0

    # ==================== 核心调用 ====================

    def call(self, image_path: str, question: str, prompt_template: str = None) -> str:
        """
        调用 Qwen3.7-Plus 多模态模型获取预测答案

        参数:
            image_path: 图像文件路径
            question: 问题文本
            prompt_template: 提示词模板（可选，未使用时直接传 question）

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
            # 构建提示词
            if prompt_template:
                text_content = prompt_template.format(question=question)
            else:
                text_content = question

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": f"file://{image_path}"},
                        {"text": text_content},
                    ],
                }
            ]

            safe_print(f"[CALL] 调用模型 [{self.call_count}] | 模型: {self.model_name} | 图像: {image_name}")

            response = dashscope.MultiModalConversation.call(
                api_key=self.api_key,
                model=self.model_name,
                messages=messages,
            )

            call_latency = time.time() - call_start_time
            self.total_latency += call_latency
            self.success_count += 1

            answer = response.output.choices[0].message.content[0]["text"]
            safe_print(f"[OK] 调用成功 [{self.call_count}] | 耗时: {call_latency:.2f}秒")
            return answer

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
            "avg_latency": avg_latency,
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
