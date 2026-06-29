"""
API调用模块
封装豆包和InternVL的API调用
"""

import os
import base64
import requests
import time

from config import doubao_config, internvl_config, prompt_config
from utils import lock, safe_print


class DoubaoAPIClient:
    """豆包API客户端"""

    def __init__(self, config=None):
        self.config = config or doubao_config
        self.api_endpoint = self.config.API_ENDPOINT
        self.api_key = self.config.API_KEY
        self.model_name = self.config.MODEL_NAME
        self.timeout = self.config.TIMEOUT
        self.temperature = self.config.TEMPERATURE
        self.thinking_mode = self.config.THINKING_MODE
        self.call_count = 0
        self.success_count = 0
        self.total_latency = 0.0

    def call(self, image_path, question, prompt_template=None):
        """
        调用豆包API获取预测答案

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
            # 读取并编码图像
            with open(image_path, 'rb') as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            # 构建提示词
            template = prompt_template or prompt_config.DEFAULT_PROMPT
            formatted_prompt = template.format(question=question)

            # 构建请求数据
            data = {
                "model": self.model_name,
                "temperature": float(self.temperature),
                "messages": [{
                    "content": [
                        {"image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, "type": "image_url"},
                        {"text": formatted_prompt, "type": "text"}
                    ],
                    "role": "user"
                }],
                "thinking": {"type": self.thinking_mode}
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            safe_print(f"[CALL] 调用豆包API [{self.call_count}] | 模型: {self.model_name} | 图像: {image_name}")

            # 发送请求
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
                safe_print(f"[OK] 豆包API调用成功 [{self.call_count}] | 耗时: {call_latency:.2f}秒")

                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0].get('message', {}).get('content', '')
                return "No response content in API result"
            else:
                safe_print(f"[ERROR] 豆包API调用失败 [{self.call_count}] | 状态码: {response.status_code} | 耗时: {call_latency:.2f}秒")
                return f"API Error: {response.status_code}"

        except requests.exceptions.RequestException as e:
            call_latency = time.time() - call_start_time
            self.total_latency += call_latency
            safe_print(f"[ERROR] 豆包API请求异常 [{self.call_count}] | 图像: {image_name} | 耗时: {call_latency:.2f}秒 | 错误: {e}")
            return f"Request Error: {str(e)}"
        except Exception as e:
            call_latency = time.time() - call_start_time
            self.total_latency += call_latency
            safe_print(f"[ERROR] 处理豆包API响应时发生错误 [{self.call_count}] | 图像: {image_name} | 耗时: {call_latency:.2f}秒 | 错误: {e}")
            return f"Processing Error: {str(e)}"

    def get_stats(self):
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

    def print_stats(self):
        """打印调用统计信息"""
        stats = self.get_stats()
        safe_print("")
        safe_print("=" * 60)
        safe_print("📊 豆包API调用统计")
        safe_print("=" * 60)
        safe_print(f"   - 模型: {self.model_name}")
        safe_print(f"   - 总调用次数: {stats['call_count']}")
        safe_print(f"   - 成功次数: {stats['success_count']}")
        safe_print(f"   - 失败次数: {stats['failed_count']}")
        safe_print(f"   - 成功率: {stats['success_rate']:.2f}%")
        safe_print(f"   - 总耗时: {self.total_latency:.2f}秒")
        safe_print(f"   - 平均耗时: {stats['avg_latency']:.2f}秒/次")
        safe_print("=" * 60)


class InternVLAPIClient:
    """InternVL API客户端"""

    def __init__(self, config=None):
        self.config = config or internvl_config
        self.api_endpoint = self.config.API_ENDPOINT
        self.model_name = self.config.MODEL_NAME
        self.timeout = self.config.TIMEOUT
        self.temperature = self.config.TEMPERATURE
        self.max_tokens = self.config.MAX_TOKENS
        self.call_count = 0
        self.success_count = 0
        self.total_latency = 0.0

    def call(self, image_path, question, prompt_template=None):
        """
        调用InternVL API获取预测答案

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
            # 读取并编码图像
            with open(image_path, 'rb') as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            # 构建提示词
            template = prompt_template or prompt_config.DEFAULT_PROMPT
            formatted_prompt = template.format(question=question)

            # 构建请求数据
            data = {
                "model": self.model_name,
                "temperature": float(self.temperature),
                "max_tokens": self.max_tokens,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, "type": "image_url"},
                        {"text": formatted_prompt, "type": "text"}
                    ]
                }]
            }

            headers = {"Content-Type": "application/json"}

            safe_print(f"[CALL] 调用InternVL API [{self.call_count}] | 模型: {self.model_name} | 图像: {image_name}")

            # 发送请求
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
                safe_print(f"[OK] InternVL API调用成功 [{self.call_count}] | 耗时: {call_latency:.2f}秒")

                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0].get('message', {}).get('content', '')
                return "No response content in API result"
            else:
                safe_print(f"[ERROR] InternVL API调用失败 [{self.call_count}] | 状态码: {response.status_code} | 耗时: {call_latency:.2f}秒")
                return f"API Error: {response.status_code}"

        except requests.exceptions.RequestException as e:
            call_latency = time.time() - call_start_time
            self.total_latency += call_latency
            safe_print(f"[ERROR] InternVL API请求异常 [{self.call_count}] | 图像: {image_name} | 耗时: {call_latency:.2f}秒 | 错误: {e}")
            return f"Request Error: {str(e)}"
        except Exception as e:
            call_latency = time.time() - call_start_time
            self.total_latency += call_latency
            safe_print(f"[ERROR] 处理InternVL API响应时发生错误 [{self.call_count}] | 图像: {image_name} | 耗时: {call_latency:.2f}秒 | 错误: {e}")
            return f"Processing Error: {str(e)}"

    def get_stats(self):
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

    def print_stats(self):
        """打印调用统计信息"""
        stats = self.get_stats()
        safe_print("")
        safe_print("=" * 60)
        safe_print("📊 InternVL API调用统计")
        safe_print("=" * 60)
        safe_print(f"   - 模型: {self.model_name}")
        safe_print(f"   - 总调用次数: {stats['call_count']}")
        safe_print(f"   - 成功次数: {stats['success_count']}")
        safe_print(f"   - 失败次数: {stats['failed_count']}")
        safe_print(f"   - 成功率: {stats['success_rate']:.2f}%")
        safe_print(f"   - 总耗时: {self.total_latency:.2f}秒")
        safe_print(f"   - 平均耗时: {stats['avg_latency']:.2f}秒/次")
        safe_print("=" * 60)


# 全局实例
doubao_client = DoubaoAPIClient()
internvl_client = InternVLAPIClient()


def call_doubao_api(image_path, question, prompt_template=None):
    """便捷函数：调用豆包API"""
    return doubao_client.call(image_path, question, prompt_template)


def call_internvl_api(image_path, question, prompt_template=None):
    """便捷函数：调用InternVL API"""
    return internvl_client.call(image_path, question, prompt_template)
