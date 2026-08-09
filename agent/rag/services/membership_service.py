"""
隶属度混合检索服务
实现 BaseRetriever，源自原 rag_rscsv_service.py (RscsvService)
负责：RAG 向量检索 → 隶属度优先检索（类别中心+Softmax）→ 基础切片降级 → 结果合并
"""
from rag.base.retriever import BaseRetriever
from rag.stores.slice_store import SliceStore
from rag.builders.slice_builder import SliceBuilder
from rag.services.knowledge_service import KnowledgeRagService
from rag.services.slice_service import retrieve_basic_slices
from rag.core.config import rag_config
from rag.core.exact_index import ExactVectorIndex
from model.factory import huggingface_embed_model
from utils.logger_handler import logger
import time
import numpy as np
from dataclasses import dataclass


def _softmax_membership(r: np.ndarray, temperature: float) -> np.ndarray:
    """
    步骤2+3: query 与各类别中心余弦 → 原始得分 r_C；Softmax(r/τ) 映射为隶属度 μ_C
    μ_C ∈ [0,1] 且 Σμ = 1；τ 越小分布越尖锐
    """
    x = np.asarray(r, dtype=np.float64) / temperature
    x = x - np.max(x)  # 数值稳定（max 平移不改变 softmax 结果）
    e = np.exp(x)
    return e / e.sum()


def retrieve_membership_slices(slice_index, store, query: str, slice_k: int, qe=None,
                               fit_threshold=None, temperature=None) -> dict:
    """
    隶属度检索内核（类别中心平均 + Softmax 映射，算法唯一实现点）
    供 MembershipHybridService 阶段1 使用；后续改造隶属度算法只需修改本函数

    算法：
      1. 类别中心：每类切片向量平均 → 归一化（版本绑定缓存，切片库手动更新后自动重算）
      2. query 与各中心余弦相似度 → 原始得分 r_C
      3. Softmax(r/τ) 归一化为隶属度 μ_C（Σμ=1），τ 取 temperature
      4. μ_C ≥ fit_threshold 的类别构成归属集合 S
      5. S 内全部切片按与 query 的余弦相似度取 top slice_k，不足 slice_k 全给
      S 为空集 → hit=False（降级由调用方编排决定，本函数不内置降级）

    Args:
        slice_index: ExactVectorIndex 全量精确检索器
        store: SliceStore（按 id 精确取文档内容）
        query: 查询文本
        slice_k: 归属类内返回的切片数量（检索多少就送多少，全部保留）
        qe: 预嵌入的 query 向量（复用上游结果时传入，None 时内部嵌入）
        fit_threshold: 归属阈值（None 时取 config/chroma.yml → retrieval.fit_threshold）
        temperature: Softmax 温度系数 τ（None 时取 config/chroma.yml → retrieval.temperature）

    Returns:
        dict: {
            "result_str": 格式化切片检索结果字符串（不含 SLICE_IDS 标记与 RAG 前缀）
            "slices_trace": [{slice_id, score(归一化), score_type, content}]，按归一化分数降序
            "hit": 是否命中（S 非空）
            "membership": {
                "max_membership": 最大隶属度 μ_max,
                "belong_clusters": [{cluster_id, center_sim, mu, slice_count}, ...]（S 内类别明细）
            }
            "timing": {"embed_latency", "search_latency", "get_latency", "total_latency"}（ms）
        }
    """
    # 参数解析：显式传入 > 配置默认
    fit_threshold = rag_config.fit_threshold if fit_threshold is None else fit_threshold
    temperature = rag_config.temperature if temperature is None else temperature

    # 嵌入 query
    _t0 = time.perf_counter()
    if qe is None:
        qe = huggingface_embed_model.embed_query(query)
    _t1 = time.perf_counter()

    # 类别中心归属判定（类别中心为版本绑定缓存：切片库手动更新后版本变化自动重算）
    V, ids, metas_all = slice_index.get_all()
    centers, uniq = slice_index.get_cluster_centers()
    if centers is None:
        _t2 = time.perf_counter()
        return {
            "result_str": "未检索到相关遥感问答参考资料。",
            "slices_trace": [],
            "hit": False,
            "membership": {"max_membership": 0.0, "belong_clusters": []},
            "timing": {
                "total_latency": (time.perf_counter() - _t0) * 1000,
                "embed_latency": (_t1 - _t0) * 1000,
                "search_latency": (_t2 - _t1) * 1000,
                "get_latency": 0.0,
            },
        }

    q_norm = np.asarray(qe, dtype=np.float64)
    q_norm = q_norm / (np.linalg.norm(q_norm) + 1e-9)
    r = centers @ q_norm  # (K,) query 与各类别中心的余弦相似度
    mu = _softmax_membership(r, temperature)
    belong_idx = np.flatnonzero(mu >= fit_threshold)
    _t2 = time.perf_counter()

    if not len(belong_idx):
        # 未命中（S 为空集）：不做类内检索，也不取每切片类别数组（省 get_cluster_ids）
        return {
            "result_str": "未检索到相关遥感问答参考资料。",
            "slices_trace": [],
            "hit": False,
            "membership": {"max_membership": float(mu.max()), "belong_clusters": []},
            "timing": {
                "total_latency": (time.perf_counter() - _t0) * 1000,
                "embed_latency": (_t1 - _t0) * 1000,
                "search_latency": (_t2 - _t1) * 1000,
                "get_latency": 0.0,
            },
        }

    # 命中：S 内全部切片按余弦取 top slice_k（不足全给）
    # 一次遍历同时构建归属类别明细与类内掩码（仅命中路径取每切片类别数组）
    cluster_ids = slice_index.get_cluster_ids()
    belong_clusters = []
    sel = np.zeros(len(cluster_ids), dtype=bool)
    for i in belong_idx:
        c_mask = cluster_ids == uniq[i]
        belong_clusters.append({
            "cluster_id": int(uniq[i]),
            "center_sim": float(r[i]),
            "mu": float(mu[i]),
            "slice_count": int(c_mask.sum()),
        })
        sel |= c_mask
    scores = V[sel] @ q_norm
    order = np.argsort(-scores)[:slice_k]
    slice_results = [(str(ids[sel][i]), float(scores[i])) for i in order]

    # 按 id 精确取文档内容；检索多少就送多少，全部保留
    top_ids = [h[0] for h in slice_results]
    got = store.get_by_ids(top_ids)
    _t3 = time.perf_counter()
    documents = got.get("documents") or []
    metadatas = got.get("metadatas") or []

    # 归一化相似度分数到 [0, 1]（min-max，保持原始降序）
    raw_scores = [h[1] for h in slice_results]
    min_score = min(raw_scores) if raw_scores else -1
    max_score = max(raw_scores) if raw_scores else 1

    def normalize_score(score: float) -> float:
        if max_score == min_score:
            return 0.5 if max_score > 0 else 0.0
        normalized = (score - min_score) / (max_score - min_score)
        return max(0.0, min(1.0, normalized))

    sorted_results = sorted(
        [
            (i, normalize_score(score), score)
            for i, score in enumerate(raw_scores)
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    slices_trace = [
        {
            "slice_id": (
                metadatas[i].get("slice_id", top_ids[i])
                if isinstance(metadatas[i], dict)
                else top_ids[i]
            ),
            "score": norm_score,
            "score_type": "membership",
            "content": documents[i],
        }
        for i, norm_score, _ in sorted_results
        if i < len(documents)
    ]
    content = "\n---\n".join(
        [
            f"相似度得分: {norm_score:.4f}\n{documents[i]}"
            for i, norm_score, _ in sorted_results
            if i < len(documents)
        ]
    )
    return {
        "result_str": f"【匹配语境分类】隶属度={float(mu.max()):.4f} "
                      f"(阈值 {fit_threshold:.2f})  \n"
                      f"隶属度得分: {float(mu.max()):.4f}\n{content}",
        "slices_trace": slices_trace,
        "hit": True,
        "membership": {"max_membership": float(mu.max()), "belong_clusters": belong_clusters},
        "timing": {
            "total_latency": (_t3 - _t0) * 1000,
            "embed_latency": (_t1 - _t0) * 1000,
            "search_latency": (_t2 - _t1) * 1000,
            "get_latency": (_t3 - _t2) * 1000,
        },
    }


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
      阶段 1: 隶属度计算（类别中心 + Softmax 映射，归属语境分类检索）
      阶段 2: fit_threshold 未命中时，降级为全量切片库检索
    """

    # ── 类级共享统计（所有实例共享，兼容旧 RscsvService 接口）──
    _total_calls: int = 0
    _hit_calls: int = 0

    # ── 会话级统计（按会话 id 隔离；UI 切会话/新会话时调用 set_current_session）──
    _session_stats: dict = {}        # session_id -> {"total_calls", "hit_calls"}
    _current_session_id: int | None = None

    def __init__(self):
        # 1. 基础构建器与切片集合
        self._builder = SliceBuilder()
        self._store = self._builder.store
        self._fit_threshold = rag_config.fit_threshold
        # 切片库全量精确检索器（numpy 暴力 top-k，替代 Chroma HNSW 近似）
        self._slice_index = ExactVectorIndex(
            collection=self._store.collection._collection,
            persist_directory=rag_config.persist_directory,
            collection_name=rag_config.slices_collection_name,
            embedding_fn=huggingface_embed_model,
        )

        # 2. 初始化知识库 RAG 总结服务
        self._knowledge_service = KnowledgeRagService()

        # 4. 配置参数
        self._slice_k = rag_config.slice_k

        # 5. RAG 知识库上下文开关（通过 chroma.yml → retrieval.enable_rag_context 控制）
        self._enable_rag_context = rag_config.enable_rag_context

        # 6. 运行时参数覆盖（UI 滑杆设置；retrieve 合并后生效，不写配置文件）
        self._runtime_params: dict = {}
        self._last_trace: RetrievalTrace | None = None

        # 7. 预热类别中心（阶段1 隶属度算法用；提前计算一次，
        #    切片库手动更新后版本变化自动重算，无需重启）
        try:
            self._slice_index.get_cluster_centers()
        except Exception as e:
            logger.warning(f"[MembershipHybridService] 类别中心预热失败，将按需计算: {e}")

    def set_runtime_params(self, **kwargs):
        """设置运行时参数覆盖（UI 在提问前调用；工具签名不变，参数经此透传）"""
        valid = {"fit_threshold", "slice_k"}
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
    def set_current_session(cls, session_id):
        """切换会话统计范围（None 表示无活动会话，统计归零显示）"""
        cls._current_session_id = session_id

    @classmethod
    def _bump_session_stats(cls, hit: bool):
        """会话级计数：一次检索调用（hit=True 表示命中）"""
        sid = cls._current_session_id
        if sid is None:
            return
        data = cls._session_stats.setdefault(sid, {"total_calls": 0, "hit_calls": 0})
        data["total_calls"] += 1
        if hit:
            data["hit_calls"] += 1

    @classmethod
    def get_session_stats_static(cls) -> dict:
        """静态方法：获取当前会话的隶属度统计（未设置会话时返回全 0）"""
        sid = cls._current_session_id
        if sid is None:
            return {"total_calls": 0, "hit_calls": 0, "hit_rate": 0.0}
        data = cls._session_stats.get(sid, {"total_calls": 0, "hit_calls": 0})
        hit_rate = data["hit_calls"] / data["total_calls"] if data["total_calls"] > 0 else 0.0
        return {"total_calls": data["total_calls"], "hit_calls": data["hit_calls"], "hit_rate": hit_rate}

    @classmethod
    def reset_membership_stats(cls):
        """重置隶属度统计数据（类级共享）"""
        cls._total_calls = 0
        cls._hit_calls = 0

    def hybrid_retrieve(
        self,
        query: str,
        slice_k: int = None,
        fit_threshold: float = None,
    ) -> str:
        """
        执行三阶段混合检索（含过程记录）

        Args:
            query: 查询文本
            slice_k / fit_threshold:
                检索参数，显式传入 > 运行时覆盖(set_runtime_params) > 配置默认
            （membership_k 已弃用：隶属度检索固定取全量库 top-1；
              top_p 已弃用：检索多少就送多少，切片检索固定保留全部 slice_k 条）

        Returns:
            格式化的检索结果字符串（契约不变）
        """
        # 参数解析：显式传入 > 运行时覆盖 > 配置默认
        slice_k = slice_k or self._resolve("slice_k", self._slice_k)
        fit_threshold = fit_threshold if fit_threshold is not None else self._resolve("fit_threshold", self._fit_threshold)

        total_start = time.time()
        trace = RetrievalTrace(
            query=query,
            decision="slice_fallback",
            params_used={
                "fit_threshold": fit_threshold,
                "slice_k": slice_k,
            },
        )

        try:
            # ==========================================
            # 阶段 0: RAG 向量检索（封装在 _retrieve_rag_context，与 slice_service 实现一致）
            # ==========================================
            t0 = time.time()
            rag_context = self._retrieve_rag_context(query)
            trace.rag_context = rag_context or None
            trace.stage0_latency = (time.time() - t0) * 1000

            # ==========================================
            # 阶段 1: 隶属度计算（封装在 _retrieve_by_membership，类别中心+Softmax 算法）
            # 仅嵌入一次 query：命中/未命中两条路复用同一向量，避免全局嵌入锁排队翻倍
            # ==========================================
            MembershipHybridService._total_calls += 1
            MembershipHybridService._bump_session_stats(hit=False)
            t1 = time.time()
            qe = self._embed_query(query)
            result_str, membership_trace = self._retrieve_by_membership(
                query, fit_threshold, slice_k, qe=qe
            )
            trace.membership = membership_trace
            trace.stage1_latency = (time.time() - t1) * 1000

            if result_str is not None:
                trace.decision = "membership_hit"
                trace.slices = (membership_trace or {}).get("final_slices", [])
                trace.total_latency = (time.time() - total_start) * 1000
                self._last_trace = trace
                return self._compose_result(rag_context, result_str)

            if membership_trace is None:
                trace.error = "隶属度计算过程异常，已降级到基础检索"

            # ==========================================
            # 阶段 2: 基础切片检索（封装在 _retrieve_by_similarity，降级回退，复用阶段1的 query 嵌入）
            # ==========================================
            t2 = time.time()
            rscsv_result, slices_trace = self._retrieve_by_similarity(query, slice_k, qe=qe)
            trace.slices = slices_trace
            trace.stage2_latency = (time.time() - t2) * 1000
            trace.total_latency = (time.time() - total_start) * 1000
            self._last_trace = trace
            return self._compose_result(rag_context, rscsv_result)
        except Exception as e:
            logger.error(f"[hybrid_retrieve] 检索过程异常: {e}", exc_info=True)
            trace.error = str(e)
            trace.total_latency = (time.time() - total_start) * 1000
            self._last_trace = trace
            raise

    def retrieve(self, query: str) -> str:
        """实现 BaseRetriever 接口（参数经 set_runtime_params 运行时配置）"""
        return self.hybrid_retrieve(query)

    def _retrieve_rag_context(self, query: str) -> str:
        """
        阶段 0: RAG 向量检索（与 slice_service._retrieve_rag_context 实现一致）
        通过 chroma.yml → retrieval.enable_rag_context 控制开关；
        封装知识库服务的上下文检索（检索 → 格式化为参考资料字符串）
        """
        return self._knowledge_service.retrieve_context(query) if self._enable_rag_context else ""

    def _embed_query(self, query: str):
        """嵌入 query 向量（阶段1/2 共用一次，避免全局嵌入锁排队翻倍）"""
        return huggingface_embed_model.embed_query(query)

    def _compose_result(self, rag_context: str, body: str) -> str:
        """拼接最终检索结果字符串：RAG 参考上下文在前，检索主体在后"""
        if rag_context:
            return f"【RAG检索参考】\n{rag_context}\n\n==============================\n\n{body}"
        return body

    def _retrieve_by_membership(
        self, query: str, fit_threshold: float,
        slice_k: int, qe=None, temperature=None,
    ) -> tuple[str | None, dict | None]:
        """
        阶段 1: 隶属度检索（类别中心平均 + Softmax 映射，统一内核 retrieve_membership_slices）
        命中：μ_max ≥ fit_threshold（归属类非空）→ 返回归属类内 top slice_k 切片内容
        未命中/异常：返回 (None, trace_data)，由编排层降级到阶段 2 全量切片检索

        qe: 预嵌入的 query 向量（复用，避免全局嵌入锁排队）
        Returns: (检索结果字符串或 None, 过程记录 dict 或 None)
        """
        trace_data = None
        try:
            data = retrieve_membership_slices(
                self._slice_index, self._store, query, slice_k,
                qe=qe, fit_threshold=fit_threshold, temperature=temperature,
            )
            membership = data["membership"]
            max_membership = membership["max_membership"]
            belong_clusters = membership["belong_clusters"]

            # trace 字段与 UI 契约对齐：qualified_log_count 语义变为归属类别数
            trace_data = {
                "max_membership": max_membership,
                "qualified_log_count": len(belong_clusters),
                "top_logs": belong_clusters,  # 类别明细（替代原日志明细）
                "final_slices": data["slices_trace"] if data["hit"] else [],
            }

            if data["hit"]:
                MembershipHybridService._hit_calls += 1
                MembershipHybridService._bump_session_stats(hit=True)
                logger.info(
                    f"【隶属度命中】max μ={max_membership:.4f} ≥ {fit_threshold:.2f}，"
                    f"归属 {len(belong_clusters)} 个语境分类"
                )
                return data["result_str"], trace_data

            logger.info(
                f"【隶属度未命中/未达标】max μ={max_membership:.4f} < "
                f"{fit_threshold:.2f}，降级到全量切片检索"
            )

        except Exception as e:
            logger.error(f"隶属度计算过程发生异常，降级到基础检索: {str(e)}")

        return None, trace_data

    def _retrieve_by_similarity(self, query: str, slice_k: int, qe=None) -> tuple[str, list]:
        """阶段 2: 全量切片库检索（阶段1 fit_threshold 未命中时降级）

        与 slice_service 阶段 1 共用同一检索内核（全量精确检索）；
        qe: 预嵌入的 query 向量（复用阶段1结果，None 时由内核嵌入）
        检索多少就送多少：slice_k 条候选全部保留
        """
        data = retrieve_basic_slices(self._slice_index, self._store, query, slice_k, qe=qe)
        return data["result_str"], data["slices_trace"]
