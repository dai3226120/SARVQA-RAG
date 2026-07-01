"""
模型基类
提取 Doubao 和 InternVL API 客户端的公共逻辑
"""

import os
import base64
import time
import requests
from abc import ABC, abstractmethod

from config import prompt_config
from utils.print_utils import safe_print


class BaseAPIClient(ABC):
    """API 客户端基类，使用模板方法模式"""

    def __init__(self, config):
        self.config = config
        self.api_endpoint = self.config.API_ENDPOINT
        self.model_name = self.config.MODEL_NAME
        self.timeout = self.config.TIMEOUT
        self.temperature = self.config.TEMPERATURE
        self.call_count = 0
        self.success_count = 0
        self.total_latency = 0.0

    # ==================== 子类必须实现 ====================

    @abstractmethod
    def _build_request_payload(self, base64_image: str, formatted_prompt: str) -> dict:
        """构建请求体"""
        pass

    @abstractmethod
    def _build_headers(self) -> dict:
        """构建请求头"""
        pass

    # ==================== 公共方法 ====================

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

    @staticmethod
    def _parse_response(result: dict) -> str:
        """解析 API 响应"""
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0].get('message', {}).get('content', '')
        return "No response content in API result"

    def call(self, image_path: str, question: str, prompt_template: str = None) -> str:
        """
        调用 API 获取预测答案

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

            data = self._build_request_payload(base64_image, formatted_prompt)
            headers = self._build_headers()

            safe_print(f"[CALL] 调用 API [{self.call_count}] | 模型: {self.model_name} | 图像: {image_name}")

            response = requests.post(
                self.api_endpoint,
                json=data,
                headers=headers,
                timeout=self.timeout
            )

            call_latency = time.time() - call_start_time
            self.total_latency += call_latency

            if response.status_code == 200:
                result = response.json()
                self.success_count += 1
                safe_print(f"[OK] API 调用成功 [{self.call_count}] | 耗时: {call_latency:.2f}秒")
                return self._parse_response(result)
            else:
                safe_print(f"[ERROR] API 调用失败 [{self.call_count}] | 状态码: {response.status_code} | 耗时: {call_latency:.2f}秒")
                return f"API Error: {response.status_code}"

        except requests.exceptions.RequestException as e:
            call_latency = time.time() - call_start_time
            self.total_latency += call_latency
            safe_print(f"[ERROR] API 请求异常 [{self.call_count}] | 图像: {image_name} | 耗时: {call_latency:.2f}秒 | 错误: {e}")
            return f"Request Error: {str(e)}"
        except Exception as e:
            call_latency = time.time() - call_start_time
            self.total_latency += call_latency
            safe_print(f"[ERROR] 处理 API 响应时发生错误 [{self.call_count}] | 图像: {image_name} | 耗时: {call_latency:.2f}秒 | 错误: {e}")
            return f"Processing Error: {str(e)}"

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
        label = label or self.model_name
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
