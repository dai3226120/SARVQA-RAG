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


# 图像识别模型工厂：用于创建图像识别模型
class DoubaoSeed20MiniModelFactory(BaseModelFactory):
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        """
        生成图像识别模型实例
        从配置文件读取模型名称
        """
        return ChatOpenAI(
            model_name=model_conf['doubao-seed-2-0-mini_model_name'],
            api_key="e9f2dacd-d0a2-4c9a-ba2a-805ee0b40dcd",  # 如果需要，可以在这里设置 API 密钥
            base_url="https://ark.cn-beijing.volces.com/api/v3",  # 如果需要，可以在这里设置 API 基础 URL
            temperature=0.7,
            # max_tokens=2048
        )


# 图像识别模型工厂：用于创建InternVL2-8B模型
class InternVL2ModelFactory(BaseModelFactory):
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatOpenAI(
            model_name=model_conf['internvl2-8b_model_name'],
            api_key="vllm-dummy-api-key",
            base_url="http://i-2.gpushare.com:24667/v1",
            # max_tokens=4096,
            temperature=0.2,
            streaming=True,
        )

# 图像识别模型工厂：用于创建InternVL3_5-8B模型
class InternVL35ModelFactory(BaseModelFactory):
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatOpenAI(
            model_name=model_conf['internvl3-5-8b_model_name'],
            api_key="vllm-dummy-api-key",
            base_url="http://i-2.gpushare.com:24667/v1",
            temperature=0.2,
            streaming=True,
        )

# 在模块加载时创建模型实例，供其他模块使用
chat_model = ChatModelFactory().generate()        # 通义千问聊天模型
embed_model = EmbeddingsFactory().generate()      # DashScope 嵌入模型
huggingface_embed_model = HuggingFaceEmbeddingsFactory().generate()  # HuggingFace 嵌入模型
doubao_seed_20_mini_model = DoubaoSeed20MiniModelFactory().generate()  # 多模态模型
internvl2_8b_model = InternVL2ModelFactory().generate()  # 图像识别模型
internvl3_5_8b_model = InternVL35ModelFactory().generate()  # 图像识别模型

