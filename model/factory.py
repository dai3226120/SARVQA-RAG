import os
from abc import ABC, abstractmethod
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
        从配置文件读取模型名称
        """
        return HuggingFaceEmbeddings(model_name=model_conf['huggingface_embedding_model_name'])


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
            extra_body={"thinking": {"type": model_conf['doubao_seed_thinking_mode']}},
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

# 在模块加载时创建模型实例，供其他模块使用
chat_model = ChatModelFactory().generate()        # 通义千问聊天模型
embed_model = EmbeddingsFactory().generate()      # DashScope 嵌入模型
huggingface_embed_model = HuggingFaceEmbeddingsFactory().generate()  # HuggingFace 嵌入模型
doubao_seed_20_mini_model = DoubaoSeed20MiniModelFactory().generate()  # 多模态模型
internvl2_8b_model = InternVL2ModelFactory().generate()  # 图像识别模型
internvl3_5_8b_model = InternVL35ModelFactory().generate()  # 图像识别模型
