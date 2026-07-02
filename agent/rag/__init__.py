"""
RAG 子系统公开 API
提供便捷的 top-level re-export，减少外部模块的导入路径长度
"""
# ── 检索服务 ──
from rag.services.knowledge_service import KnowledgeRagService
from rag.services.slice_service import SliceRetrievalService
from rag.services.membership_service import MembershipHybridService
from rag.services.stats import HitRateTracker

# ── 构建器 ──
from rag.builders.knowledge_builder import KnowledgeBuilder
from rag.builders.slice_builder import SliceBuilder

# ── 向量库 ──
from rag.stores.knowledge_store import KnowledgeStore
from rag.stores.slice_store import SliceStore

# ── 隶属度/缓存系统 ──
from rag.membership.cache_system import SemanticCacheSystem
from rag.membership.degree_calculator import MembershipCalculator
from rag.membership.vlm_evaluator import VlmEvaluator
from rag.membership.llm_judge import LlmJudge
from rag.membership.log_manager import LogManager

# ── 抽象基类 ──
from rag.base.vector_store import BaseVectorStore
from rag.base.retriever import BaseRetriever
from rag.base.llm_judge_client import LLMJudgeClient

# ── 共享基础设施 ──
from rag.core.config import rag_config, RagConfig
from rag.core.md5_store import Md5Store
from rag.core.chroma_manager import ChromaManager

# ── 评估 ──
from rag.eval.real_rag_engine import RealRAGEngine

__all__ = [
    # 服务
    "KnowledgeRagService",
    "SliceRetrievalService",
    "MembershipHybridService",
    "HitRateTracker",
    # 构建器
    "KnowledgeBuilder",
    "SliceBuilder",
    # 向量库
    "KnowledgeStore",
    "SliceStore",
    # 隶属度/缓存
    "SemanticCacheSystem",
    "MembershipCalculator",
    "VlmEvaluator",
    "LlmJudge",
    "LogManager",
    # 基类
    "BaseVectorStore",
    "BaseRetriever",
    "LLMJudgeClient",
    # 共享
    "rag_config",
    "RagConfig",
    "Md5Store",
    "ChromaManager",
    # 评估
    "RealRAGEngine",
]
