"""
隶属度混合检索服务
实现 BaseRetriever，源自原 rag_rscsv_service.py (RscsvService)
负责：RAG 向量检索 → 隶属度优先检索 → 基础切片降级 → 结果合并
"""
from rag.base.retriever import BaseRetriever
from rag.stores.slice_store import SliceStore
from rag.builders.slice_builder import SliceBuilder
from rag.services.knowledge_service import KnowledgeRagService
from rag.membership.cache_system import SemanticCacheSystem
from rag.core.config import rag_config
from utils.logger_handler import logger


class MembershipHybridService(BaseRetriever):
    """
    隶属度混合检索服务
    三阶段检索：
      阶段 0: RAG 向量检索（通用知识库）
      阶段 1: 隶属度计算（日志缓存 + 阈值校验）
      阶段 2: 隶属度不达标时，降级为基础切片检索
    """

    # ── 类级共享统计（所有实例共享，兼容旧 RscsvService 接口）──
    _total_calls: int = 0
    _hit_calls: int = 0

    def __init__(self):
        # 1. 基础构建器与切片集合
        self._builder = SliceBuilder()
        self._store = self._builder.store
        self._fit_threshold = rag_config.fit_threshold

        # 2. 初始化隶属度缓存系统
        self._cache_system = SemanticCacheSystem()

        # 3. 初始化知识库 RAG 总结服务
        self._knowledge_service = KnowledgeRagService()

        # 4. 配置参数
        self._top_p = rag_config.top_p
        self._slice_k = rag_config.slice_k
        self._membership_k = rag_config.membership_k

        # 5. RAG 知识库上下文开关（通过 chroma.yml → retrieval.enable_rag_context 控制）
        self._enable_rag_context = rag_config.enable_rag_context

    def get_membership_stats(self) -> dict:
        """获取隶属度统计信息（委托到类级共享记录）"""
        hit_rate = (
            MembershipHybridService._hit_calls / MembershipHybridService._total_calls
            if MembershipHybridService._total_calls > 0
            else 0.0
        )
        return {
            "total_calls": MembershipHybridService._total_calls,
            "hit_calls": MembershipHybridService._hit_calls,
            "hit_rate": hit_rate,
        }

    @classmethod
    def get_membership_stats_static(cls) -> dict:
        """静态方法：获取隶属度统计信息（类级共享，所有实例共享同一份记录）"""
        hit_rate = cls._hit_calls / cls._total_calls if cls._total_calls > 0 else 0.0
        return {"total_calls": cls._total_calls, "hit_calls": cls._hit_calls, "hit_rate": hit_rate}

    @classmethod
    def get_membership_hit_rate_static(cls) -> float:
        """静态方法：获取隶属度命中率"""
        if cls._total_calls == 0:
            return 0.0
        return cls._hit_calls / cls._total_calls

    @classmethod
    def reset_membership_stats(cls):
        """重置隶属度统计数据（类级共享）"""
        cls._total_calls = 0
        cls._hit_calls = 0

    def hybrid_retrieve(
        self,
        query: str,
        slice_k: int = None,
        membership_k: int = None,
        top_p: int = None,
        fit_threshold: float = None,
    ) -> str:
        """
        执行三阶段混合检索

        Args:
            query: 查询文本
            slice_k: 基础切片检索数量
            membership_k: 隶属度检索数量
            top_p: 保留结果数量
            fit_threshold: 隶属度阈值

        Returns:
            格式化的检索结果字符串
        """
        slice_k = slice_k or self._slice_k
        membership_k = membership_k or self._membership_k
        top_p = top_p or self._top_p
        fit_threshold = fit_threshold or self._fit_threshold

        # ==========================================
        # 阶段 0: RAG 向量检索（通过 chroma.yml → retrieval.enable_rag_context 控制）
        # ==========================================
        rag_context = self._knowledge_service.retrieve_context(query) if self._enable_rag_context else ""

        # ==========================================
        # 阶段 1: 隶属度计算（缓存拦截与校验）
        # ==========================================
        MembershipHybridService._total_calls += 1
        membership_result = self._retrieve_by_membership(
            query, membership_k, fit_threshold, top_p
        )
        if membership_result is not None:
            if rag_context:
                return f"【RAG检索参考】\n{rag_context}\n\n==============================\n\n{membership_result}"
            return membership_result

        # ==========================================
        # 阶段 2: 基础切片检索（降级回退）
        # ==========================================
        rscsv_result = self._retrieve_by_similarity(query, slice_k, top_p)
        if rag_context:
            return f"【RAG检索参考】\n{rag_context}\n\n==============================\n\n{rscsv_result}"
        return rscsv_result

    def retrieve(self, query: str) -> str:
        """实现 BaseRetriever 接口"""
        return self.hybrid_retrieve(
            query,
            slice_k=self._slice_k,
            membership_k=self._membership_k,
            top_p=self._top_p,
            fit_threshold=self._fit_threshold,
        )

    def _retrieve_by_membership(
        self, query: str, membership_k: int, fit_threshold: float, top_p: int
    ) -> str | None:
        """
        阶段 1: 隶属度检索
        Returns: 检索结果字符串，或 None 表示未命中需降级
        """
        try:
            membership_result = self._cache_system.calculate_membership_degree(
                query, k=membership_k, fit_threshold=fit_threshold, top_p=top_p
            )
            max_membership = membership_result.get("max_membership", 0.0)
            qualified_log_count = membership_result.get("qualified_log_count", 0)
            qualified_memberships = membership_result.get("qualified_memberships", [])

            memberships_str = ", ".join([f"{m:.4f}" for m in qualified_memberships])

            if qualified_log_count > 0 and membership_result.get("weighted_slices"):
                logger.info(
                    f"【隶属度命中】合格日志: {qualified_log_count}条，"
                    f"隶属度(从大到小): [{memberships_str}]"
                )

                top_slice_info = membership_result["weighted_slices"][:top_p]
                top_slice_ids = [s["slice_id"] for s in top_slice_info]
                logger.info(f" 正在尝试从切片库获取以下 ID 的数据: {top_slice_ids}")

                slices_data = self._store.get_by_ids(top_slice_ids)
                documents = slices_data.get("documents", [])
                metadatas = slices_data.get("metadatas", [])

                logger.info(f" 切片库实际返回了 {len(documents)} 条文档内容")

                if documents:
                    MembershipHybridService._hit_calls += 1

                    # 按隶属度得分排序
                    doc_with_membership = []
                    for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
                        slice_id = (
                            metadata.get("slice_id")
                            if isinstance(metadata, dict)
                            else top_slice_ids[i]
                        )
                        membership_degree = next(
                            (
                                s["membership_degree"]
                                for s in top_slice_info
                                if s["slice_id"] == slice_id
                            ),
                            0.0,
                        )
                        doc_with_membership.append((doc, membership_degree))

                    doc_with_membership.sort(key=lambda x: x[1], reverse=True)

                    content = "\n---\n".join(
                        [
                            f"隶属度得分: {score:.4f}\n{doc}"
                            for doc, score in doc_with_membership
                        ]
                    )
                    return (
                        f"【匹配隶属度缓存】合格日志={qualified_log_count}条，"
                        f"隶属度(从大到小): [{memberships_str}] "
                        f"(共检索{membership_k}条，"
                        f"按隶属度排序后保留{len(doc_with_membership)}条)  \n{content}"
                    )
            else:
                logger.info(
                    f"【隶属度未命中/未达标】合格日志: {qualified_log_count}条，"
                    f"最大隶属度: {max_membership:.4f}"
                )

        except Exception as e:
            logger.error(f"隶属度计算过程发生异常，降级到基础检索: {str(e)}")

        return None

    def _retrieve_by_similarity(self, query: str, slice_k: int, top_p: int) -> str:
        """阶段 2: 基础切片检索（降级回退）"""
        slice_results = self._store.similarity_search_with_scores(query, k=slice_k)
        if slice_results:
            sorted_results = sorted(slice_results, key=lambda x: x[1], reverse=True)
            top_results = sorted_results[:top_p]

            content = "\n---\n".join(
                [
                    f"相似度得分: {score:.4f}\n{doc.page_content}"
                    for doc, score in top_results
                ]
            )
            return (
                f"【匹配基础切片】(共检索{slice_k}条，"
                f"按相似度排序后保留{len(top_results)}条)  \n{content}"
            )

        return "未检索到相关遥感问答参考资料。"
