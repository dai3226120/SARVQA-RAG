import os
import threading
# Windows 下 pyarrow(pandas 间接依赖) 先加载会导致 ortools DLL 加载失败(WinError 127)。
# 各入口脚本已在 pandas 之前导入 k_means_constrained；此处再兜底一次，
# 保证本模块(权重加载源头)导入时 ortools 扩展 DLL 已在进程内。
try:
    import k_means_constrained  # noqa: F401
except ImportError:
    pass  # 无该依赖时跳过（仅失去顺序保护）
from abc import ABC, abstractmethod


class _ThreadSafeEmbeddings:
    """线程安全包装：GPU 嵌入推理本质串行，并发调用会导致严重退化
    （实测 6 并发下单调用 15~25ms → 110ms），加全局锁让调用排队而非退化。"""

    def __init__(self, inner):
        self._inner = inner
        self._lock = threading.Lock()

    def embed_query(self, text):
        with self._lock:
            return self._inner.embed_query(text)

    def embed_documents(self, texts):
        with self._lock:
            return self._inner.embed_documents(texts)

    def __getattr__(self, name):
        return getattr(self._inner, name)
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_huggingface import HuggingFaceEmbeddings  # HuggingFace嵌入模型
from langchain_openai import ChatOpenAI
from utils.config_handler import model_conf


# 抽象基类：模型工厂的父类，定义统一接口
# 所有具体模型工厂必须实现 generate 方法
class BaseModelFactory(ABC):
    @abstractmethod
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        """
        生成并返回模型实例
        返回值可以是 Embeddings（嵌入模型）或 BaseChatModel（聊天模型）
        """
        pass


# 聊天模型工厂：用于创建通义千问聊天模型
class ChatModelFactory(BaseModelFactory):
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        """
        生成通义千问聊天模型实例
        从配置文件读取模型名称
        """
        return ChatTongyi(model=model_conf['chat_model_name'])


# 嵌入模型工厂：用于创建 DashScope 嵌入模型
class EmbeddingsFactory(BaseModelFactory):
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        """
        生成 DashScope 嵌入模型实例
        从配置文件读取模型名称
        """
        return DashScopeEmbeddings(model=model_conf['embedding_model_name'])


# HuggingFace 嵌入模型工厂：用于创建 HuggingFace 嵌入模型
class HuggingFaceEmbeddingsFactory(BaseModelFactory):
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        """
        生成 HuggingFace 嵌入模型实例
        从配置文件读取模型名称；推理设备优先取配置（huggingface_embedding_device），
        未配置或为 auto 时自动检测 CUDA——有 GPU 用 GPU 推理，否则回落 CPU
        """
        device = model_conf.get('huggingface_embedding_device', 'auto')
        if device == 'auto':
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return HuggingFaceEmbeddings(
            model_name=model_conf['huggingface_embedding_model_name'],
            model_kwargs={"device": device},
        )


# 图像识别模型工厂：用于创建 Doubao Seed 模型
class DoubaoSeed20MiniModelFactory(BaseModelFactory):
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        """
        生成 Doubao Seed 模型实例
        API key 和 temperature 优先从环境变量读取，兼容 .env 配置
        thinking 参数通过 extra_body 注入，兼容 OpenAI client
        """
        return ChatOpenAI(
            model_name=model_conf['doubao-seed-2-0-mini_model_name'],
            api_key=os.environ.get('DOUBAO_SEED_API_KEY') or model_conf.get('doubao_seed_api_key', ''),
            base_url=model_conf['doubao_seed_api_endpoint'],
            temperature=float(os.environ.get('DOUBAO_SEED_TEMPERATURE') or model_conf.get('doubao_seed_temperature', 0.7)),
            # 请求超时：API 挂起时终止等待（否则单条挂起会冻结整个 as_completed 流水线）
            timeout=120,
            max_retries=1,
            extra_body={"thinking": {"type": model_conf['doubao_seed_thinking_mode']}},
        )


# Doubao 1.5 Lite 模型工厂：用于 LLM 语义判断（替代原原生 requests 调用）
class DoubaoLiteModelFactory(BaseModelFactory):
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        """
        生成 Doubao 1.5 Lite 模型实例
        用于语义匹配判断等轻量级 LLM 任务
        API key 优先从环境变量读取，兼容 .env 配置
        """
        return ChatOpenAI(
            model_name=model_conf['doubao-1-5-lite_model_name'],
            api_key=os.environ.get('DOUBAO_SEED_API_KEY') or model_conf.get('doubao_seed_api_key', ''),
            # 注意：ChatOpenAI 会自动追加 /chat/completions，必须用不带后缀的 api/v3 端点
            # （doubao_seed_full_endpoint 带后缀会导致路径重复，服务端持续返回 500）
            base_url=model_conf['doubao_seed_api_endpoint'],
            temperature=float(os.environ.get('DOUBAO_SEED_TEMPERATURE') or model_conf.get('doubao_1_5_lite_temperature', 0.7)),
            timeout=model_conf.get('doubao_1_5_lite_timeout', 30),
            extra_body={"thinking": {"type": model_conf.get('doubao_1_5_lite_thinking_mode', 'disabled')}},
        )


# Qwen3.7 Plus 模型工厂：用于文本问答（阿里云 MaaS，OpenAI 兼容）
class Qwen37PlusModelFactory(BaseModelFactory):
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatOpenAI(
            model_name=model_conf['qwen37_plus_model_name'],
            api_key=os.environ.get('DASHSCOPE_API_KEY') or model_conf.get('qwen37_plus_api_key', ''),
            base_url=model_conf['qwen37_plus_api_endpoint'],
            temperature=0.7,
            streaming=True,
            timeout=30,
        )


# 图像识别模型工厂：用于创建InternVL2-8B模型
class InternVL2ModelFactory(BaseModelFactory):
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatOpenAI(
            model_name=model_conf['internvl2-8b_model_name'],
            api_key=model_conf['internvl_api_key'],
            base_url=model_conf['internvl_api_endpoint'],
            temperature=model_conf['internvl_temperature'],
            streaming=True,
        )

# 图像识别模型工厂：用于创建InternVL3_5-8B模型
class InternVL35ModelFactory(BaseModelFactory):
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatOpenAI(
            model_name=model_conf['internvl3-5-8b_model_name'],
            api_key=model_conf['internvl_api_key'],
            base_url=model_conf['internvl_api_endpoint'],
            temperature=model_conf['internvl_temperature'],
            streaming=True,
            max_tokens=model_conf['internvl_max_tokens'],
        )

# vLLM/OpenAI 兼容部署模型工厂（tinygptv / tinygptv-stage4 / geochat / skyeyegpt）：
# 字段未单独配置时自动沿用 InternVL 的部署参数（端点/API Key/温度/输出上限），
# 单独配置的字段优先（如 tinygptv 的输出上限）。
# ⚠️ 改名：发布正式模型时把占位名替换为正式模型名（实例名 + 下方配置前缀）
class VLLMChatModelFactory(BaseModelFactory):
    """vLLM 部署模型工厂，按配置前缀读取 model.yml 中的模型参数"""

    def __init__(self, prefix: str):
        self._prefix = prefix

    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        # 停止符/采样参数（可选）：按前缀配置，如 tinygptv_stop: ["###"]、
        # tinygptv_repetition_penalty: 1.1（vLLM 专有参数经 extra_body 透传）；
        # 未配置的模型（model-2/3 等）不带这些参数，行为不受影响
        stop = model_conf.get(f'{self._prefix}_stop')
        top_p = model_conf.get(f'{self._prefix}_top_p')
        extra_body = {}
        rep = model_conf.get(f'{self._prefix}_repetition_penalty')
        if rep is not None:
            extra_body["repetition_penalty"] = float(rep)
        return ChatOpenAI(
            model_name=model_conf[f'{self._prefix}_model_name'],
            api_key=model_conf.get(f'{self._prefix}_api_key', model_conf['internvl_api_key']),
            base_url=model_conf.get(f'{self._prefix}_api_endpoint', model_conf['internvl_api_endpoint']),
            temperature=float(model_conf.get(f'{self._prefix}_temperature', model_conf['internvl_temperature'])),
            streaming=True,
            max_tokens=model_conf.get(f'{self._prefix}_max_tokens', model_conf['internvl_max_tokens']),
            stop=stop,
            top_p=top_p,
            extra_body=extra_body or None,
        )


# 在模块加载时创建模型实例，供其他模块使用
chat_model = ChatModelFactory().generate()        # 通义千问聊天模型
embed_model = EmbeddingsFactory().generate()      # DashScope 嵌入模型
huggingface_embed_model = _ThreadSafeEmbeddings(  # 线程安全包装：消除 GPU 嵌入并发退化
    HuggingFaceEmbeddingsFactory().generate()
)  # HuggingFace 嵌入模型
doubao_seed_20_mini_model = DoubaoSeed20MiniModelFactory().generate()  # 多模态模型
internvl2_8b_model = InternVL2ModelFactory().generate()  # 图像识别模型
internvl3_5_8b_model = InternVL35ModelFactory().generate()  # 图像识别模型
tinygptv_model = VLLMChatModelFactory("tinygptv").generate()  # TinyGPT-V/SAR-GPT（vLLM）
tinygptv_stage4_model = VLLMChatModelFactory("tinygptv_stage4").generate()  # 官方 Stage4（[INST] 模板）
geochat_model = VLLMChatModelFactory("geochat").generate()    # GeoChat-7B（LLaVA-1.5，vLLM）
skyeyegpt_model = VLLMChatModelFactory("skyeyegpt").generate()  # SkyEyeGPT（MiniGPT-v2 架构，OpenAI 兼容服务端）
imagerag_model = VLLMChatModelFactory("imagerag").generate()    # ImageRAG（InternVL2.5-8B+LoRA，vLLM）
doubao_1_5_lite_model = DoubaoLiteModelFactory().generate()  # LLM 语义判断模型
qwen37_plus_model = Qwen37PlusModelFactory().generate()  # Qwen3.7 Plus 文本模型
