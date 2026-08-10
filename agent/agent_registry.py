"""Agent 模型注册表：模型 key → 代理类/模型实例/能力分组

供 mainapp.py 模型切换下拉使用；build_agent 按 key 创建 agent 实例。
"""
from model.factory import (
    chat_model,
    doubao_seed_20_mini_model,
    internvl2_8b_model,
    internvl3_5_8b_model,
    doubao_1_5_lite_model,
    qwen37_plus_model,
    tinygptv_model,
)
from mainagent import MainAgent
from mainagent_internVL import MainAgent as MainAgentInternVL

MODEL_REGISTRY = {
    # ── 多模态 VLM ──
    "doubao-seed-2.0-mini": {
        "class": MainAgent, "model": doubao_seed_20_mini_model,
        "supports_image": True, "group": "多模态", "label": "Doubao Seed 2.0 Mini",
    },
    "internvl3.5-8b": {
        "class": MainAgentInternVL, "model": internvl3_5_8b_model,
        "supports_image": True, "group": "多模态", "label": "InternVL3.5-8B",
    },
    "internvl2-8b": {
        "class": MainAgentInternVL, "model": internvl2_8b_model,
        "supports_image": True, "group": "多模态", "label": "InternVL2-8B",
    },
    "tinygptv": {
        "class": MainAgentInternVL, "model": tinygptv_model,
        "supports_image": True, "group": "多模态", "label": "TinyGPT-V",
    },
    # ── 文本 LLM ──
    "doubao-1.5-lite": {
        "class": MainAgent, "model": doubao_1_5_lite_model,
        "supports_image": False, "group": "文本", "label": "Doubao 1.5 Lite",
    },
    "qwen3-max": {
        "class": MainAgent, "model": chat_model,
        "supports_image": False, "group": "文本", "label": "Qwen3 Max",
    },
    "qwen3.7-plus": {
        "class": MainAgent, "model": qwen37_plus_model,
        "supports_image": False, "group": "文本", "label": "Qwen3.7 Plus",
    },
}

DEFAULT_MODEL_KEY = "doubao-seed-2.0-mini"   # 多模态默认
DEFAULT_TEXT_MODEL_KEY = "doubao-1.5-lite"   # 文本默认


def build_agent(model_key: str):
    """按 key 创建 agent 实例（internvl 两步式需指定 vision_model）"""
    info = MODEL_REGISTRY[model_key]
    if info["class"] is MainAgentInternVL:
        return info["class"](vision_model=info["model"])
    return info["class"](model=info["model"])
