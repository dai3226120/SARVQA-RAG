"""
Doubao（豆包）模型 API 客户端
"""

from .base_model import BaseAPIClient
from config import doubao_config


class DoubaoAPIClient(BaseAPIClient):
    """豆包 API 客户端"""

    def __init__(self, config=None):
        super().__init__(config or doubao_config)
        self.thinking_mode = self.config.THINKING_MODE

    def _build_request_payload(self, base64_image: str, formatted_prompt: str) -> dict:
        return {
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

    def _build_headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.API_KEY}"
        }

    def print_stats(self):
        """打印调用统计信息"""
        super().print_stats(label="豆包")
