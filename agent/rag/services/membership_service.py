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
import time
from dataclasses import dataclass


@dataclass
class RetrievalTrace:
    """一次 hybrid_retrieve 调用的完整过程记录（供 UI 展示，工具字符串不受影响）"""

    query: str
    decision: str                      # "membership_hit" | "slice_fallback"
    params_used: dict                  # 实际使用的检索参数
    rag_context: str | None = None     # 阶段0
    stage0_latency: float = 0.0
    membership: dict | None = None     # 阶段1: max_membership / qualified_log_count / top_logs / final_slices
    stage1_latency: float = 0.0
    slices: list | None = None         # 最终切片: [{slice_id, score, score_type, content}]
    stage2_latency: float = 0.0
    total_latency: float = 0.0
    error: str | None = None


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

        # 6. 运行时参数覆盖（UI 滑杆设置；retrieve 合并后生效，不写配置文件）
        self._runtime_params: dict = {}
        self._last_trace: RetrievalTrace | None = None

    def set_runtime_params(self, **kwargs):
        """设置运行时参数覆盖（UI 在提问前调用；工具签名不变，参数经此透传）"""
        valid = {"w1", "w2", "fit_threshold", "membership_k", "slice_k", "top_p"}
        self._runtime_params = {k: v for k, v in kwargs.items() if k in valid and v is not None}

    def clear_runtime_params(self):
        """清除运行时参数覆盖，回落配置文件默认值"""
        self._runtime_params = {}

    def _resolve(self, key: str, default):
        """参数解析：运行时覆盖 > 默认值"""
        return self._runtime_params.get(key, default)

    def get_last_trace(self) -> RetrievalTrace | None:
        """获取最近一次检索的过程记录（供 UI 展示）"""
        return self._last_trace

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
        w1: float = None,
        w2: float = None,
    ) -> str:
        """
        执行三阶段混合检索（含过程记录）

        Args:
            query: 查询文本
            slice_k / membership_k / top_p / fit_threshold / w1 / w2:
                检索参数，显式传入 > 运行时覆盖(set_runtime_params) > 配置默认

        Returns:
            格式化的检索结果字符串（契约不变）
        """
        # 参数解析：显式传入 > 运行时覆盖 > 配置默认
        slice_k = slice_k or self._resolve("slice_k", self._slice_k)
        membership_k = membership_k or self._resolve("membership_k", self._membership_k)
        top_p = top_p or self._resolve("top_p", self._top_p)
        fit_threshold = fit_threshold if fit_threshold is not None else self._resolve("fit_threshold", self._fit_threshold)
        w1 = w1 if w1 is not None else self._resolve("w1", rag_config.w1)
        w2 = w2 if w2 is not None else self._resolve("w2", rag_config.w2)

        total_start = time.time()
        trace = RetrievalTrace(
            query=query,
            decision="slice_fallback",
            params_used={
                "w1": w1, "w2": w2, "fit_threshold": fit_threshold,
                "membership_k": membership_k, "slice_k": slice_k, "top_p": top_p,
            },
        )

        try:
            # ==========================================
            # 阶段 0: RAG 向量检索（通过 chroma.yml → retrieval.enable_rag_context 控制）
            # ==========================================
            t0 = time.time()
            rag_context = self._knowledge_service.retrieve_context(query) if self._enable_rag_context else ""
            trace.rag_context = rag_context or None
            trace.stage0_latency = (time.time() - t0) * 1000

            # ==========================================
            # 阶段 1: 隶属度计算（缓存拦截与校验）
            # ==========================================
            MembershipHybridService._total_calls += 1
            t1 = time.time()
            result_str, membership_trace = self._retrieve_by_membership(
                query, membership_k, fit_threshold, top_p, w1, w2
            )
            trace.membership = membership_trace
            trace.stage1_latency = (time.time() - t1) * 1000

            if result_str is not None:
                trace.decision = "membership_hit"
                trace.slices = (membership_trace or {}).get("final_slices", [])
                trace.total_latency = (time.time() - total_start) * 1000
                self._last_trace = trace
                if rag_context:
                    return f"【RAG检索参考】\n{rag_context}\n\n==============================\n\n{result_str}"
                return result_str

            if membership_trace is None:
                trace.error = "隶属度计算过程异常，已降级到基础检索"

            # ==========================================
            # 阶段 2: 基础切片检索（降级回退）
            # ==========================================
            t2 = time.time()
            rscsv_result, slices_trace = self._retrieve_by_similarity(query, slice_k, top_p)
            trace.slices = slices_trace
            trace.stage2_latency = (time.time() - t2) * 1000
            trace.total_latency = (time.time() - total_start) * 1000
            self._last_trace = trace
            if rag_context:
                return f"【RAG检索参考】\n{rag_context}\n\n==============================\n\n{rscsv_result}"
            return rscsv_result
        except Exception as e:
            logger.error(f"[hybrid_retrieve] 检索过程异常: {e}", exc_info=True)
            trace.error = str(e)
            trace.total_latency = (time.time() - total_start) * 1000
            self._last_trace = trace
            raise

    def retrieve(self, query: str) -> str:
        """实现 BaseRetriever 接口（参数经 set_runtime_params 运行时配置）"""
        return self.hybrid_retrieve(query)

    def _retrieve_by_membership(
        self, query: str, membership_k: int, fit_threshold: float, top_p: int,
        w1: float, w2: float,
    ) -> tuple[str | None, dict | None]:
        """
        阶段 1: 隶属度检索
        Returns: (检索结果字符串或 None, 过程记录 dict 或 None)
        """
        trace_data = None
        try:
            membership_result = self._cache_system.calculate_membership_degree(
                query, k=membership_k, fit_threshold=fit_threshold, top_p=top_p, w1=w1, w2=w2
            )
            max_membership = membership_result.get("max_membership", 0.0)
            qualified_log_count = membership_result.get("qualified_log_count", 0)
            qualified_memberships = membership_result.get("qualified_memberships", [])
            top_logs = membership_result.get("top_logs", [])

            memberships_str = ", ".join([f"{m:.4f}" for m in qualified_memberships])

            trace_data = {
                "max_membership": max_membership,
                "qualified_log_count": qualified_log_count,
                "qualified_memberships": qualified_memberships,
                "top_logs": top_logs,
                "final_slices": [],
            }

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

                    # 按隶属度得分排序（同时携带 slice_id 供 trace 使用）
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
                        doc_with_membership.append((doc, membership_degree, slice_id))

                    doc_with_membership.sort(key=lambda x: x[1], reverse=True)

                    trace_data["final_slices"] = [
                        {"slice_id": sid, "score": score,
                         "score_type": "membership", "content": doc}
                        for doc, score, sid in doc_with_membership
                    ]

                    content = "\n---\n".join(
                        [
                            f"隶属度得分: {score:.4f}\n{doc}"
                            for doc, score, _ in doc_with_membership
                        ]
                    )
                    return (
                        f"【匹配隶属度缓存】合格日志={qualified_log_count}条，"
                        f"隶属度(从大到小): [{memberships_str}] "
                        f"(共检索{membership_k}条，"
                        f"按隶属度排序后保留{len(doc_with_membership)}条)  \n{content}",
                        trace_data,
                    )
            else:
                logger.info(
                    f"【隶属度未命中/未达标】合格日志: {qualified_log_count}条，"
                    f"最大隶属度: {max_membership:.4f}"
                )

        except Exception as e:
            logger.error(f"隶属度计算过程发生异常，降级到基础检索: {str(e)}")

        return None, trace_data

    def _retrieve_by_similarity(self, query: str, slice_k: int, top_p: int) -> tuple[str, list]:
        """阶段 2: 基础切片检索（降级回退），返回 (结果字符串, 切片trace列表)"""
        slice_results = self._store.similarity_search_with_scores(query, k=slice_k)
        if slice_results:
            sorted_results = sorted(slice_results, key=lambda x: x[1], reverse=True)
            top_results = sorted_results[:top_p]

            slices_trace = [
                {
                    "slice_id": doc.metadata.get("slice_id", ""),
                    "score": float(score),
                    "score_type": "similarity",
                    "content": doc.page_content,
                }
                for doc, score in top_results
            ]

            content = "\n---\n".join(
                [
                    f"相似度得分: {score:.4f}\n{doc.page_content}"
                    for doc, score in top_results
                ]
            )
            return (
                f"【匹配基础切片】(共检索{slice_k}条，"
                f"按相似度排序后保留{len(top_results)}条)  \n{content}",
                slices_trace,
            )

        return "未检索到相关遥感问答参考资料。", []
