# tests/test_model_factory.py
"""模型工厂配置回归测试

背景：DoubaoLiteModelFactory 曾使用 doubao_seed_full_endpoint（带 /chat/completions 后缀），
而 ChatOpenAI 会自动追加该后缀，导致路径重复、服务端持续返回 500。
"""
import pytest

from utils.config_handler import model_conf


def test_lite_model_uses_base_endpoint_not_full():
    """回归：lite 工厂的 base_url 必须是 api/v3（不带 /chat/completions）"""
    from model.factory import DoubaoLiteModelFactory

    client = DoubaoLiteModelFactory().generate()
    base_url = client.openai_api_base.rstrip("/")
    expected = model_conf["doubao_seed_api_endpoint"].rstrip("/")
    assert base_url == expected
    assert not base_url.endswith("/chat/completions")
