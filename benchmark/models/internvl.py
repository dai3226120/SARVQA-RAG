"""
InternVL 模型 API 客户端
"""

from .base_model import BaseAPIClient
from config import internvl_config


class InternVLAPIClient(BaseAPIClient):
    """InternVL API 客户端"""

    def __init__(self, config=None):
        super().__init__(config or internvl_config)
        self.max_tokens = self.config.MAX_TOKENS

    def _build_request_payload(self, base64_image: str, formatted_prompt: str) -> dict:
        return {
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

    def _build_headers(self) -> dict:
        return {"Content-Type": "application/json"}

    def print_stats(self):
        """打印调用统计信息"""
        super().print_stats(label="InternVL")
