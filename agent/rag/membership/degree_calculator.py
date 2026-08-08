"""
隶属度计算核心模块
源自原 rag_rsfit_builder.py SARSemanticCacheSystem.calculate_membership_degree
"""
from utils.logger_handler import logger
from rag.core.config import rag_config


class MembershipCalculator:
    """
    隶属度计算器
    综合考虑：
    1. 新问题与日志问题+切片的语义相似度（全量精确检索，非 HNSW 近似）
    2. 日志中的正确性分数（作为该问题的可信度）
    """

    def __init__(
        self,
        logs_index,
        w1: float = None,
        w2: float = None,
        bleu_weight: float = None,
        overlap_weight: float = None,
    ):
        """
        Args:
            logs_index: 日志库全量精确检索器（ExactVectorIndex）
            w1: 相似度权重
            w2: 正确性分数权重
            bleu_weight: BLEU 分数权重
            overlap_weight: 词汇重叠权重
        """
        self._index = logs_index
        self._w1 = w1 if w1 is not None else rag_config.w1
        self._w2 = w2 if w2 is not None else rag_config.w2

    def calculate(
        self,
        query: str,
        fit_threshold: float = None,
        w1: float = None,
        w2: float = None,
        qe=None,
    ) -> dict:
        """
        计算新问题与日志的隶属度（全量检索 → 只取隶属度最高的一条）

        语义：日志库为全量精确检索（faiss 暴力扫描），相似度排序全局最优，
        只需召回隶属度最高的一条日志；该条 ≥ 阈值即命中，直接用其存储的
        切片内容快照召回，不做多余处理。

        Args:
            query: 新查询问题
            fit_threshold: 隶属度阈值
            w1: 相似度权重（运行时透传，优先于构造默认）
            w2: 正确性分数权重（运行时透传，优先于构造默认）

        Returns:
            dict: 包含 membership_score / max_membership / top_logs /
                  weighted_slices / qualified_log_count / qualified_memberships
        """
        fit_threshold = fit_threshold if fit_threshold is not None else rag_config.fit_threshold

        # 解析权重：显式传入 > 构造时默认 > 配置默认；和不为 1 自动归一化
        w1 = w1 if w1 is not None else self._w1
        w2 = w2 if w2 is not None else self._w2
        if abs(w1 + w2 - 1.0) > 1e-6:
            logger.warning("权重和不为1，进行自动归一化: w1=%.2f, w2=%.2f", w1, w2)
            total = w1 + w2
            if total == 0:
                logger.warning("权重和为零，回退默认权重 w1=0.5, w2=0.5")
                w1, w2 = 0.5, 0.5
            else:
                w1 = w1 / total
                w2 = w2 / total

        # 1. 全量精确检索日志库，只取隶属度最高的 1 条（qe 复用上层预嵌入，省一次全局锁排队）
        try:
            results = self._index.search_text(query, 1, qe=qe)
        except Exception as e:
            logger.error("日志库全量精确检索失败: %s", e)
            return self._empty_result()

        if not results:
            logger.warning("未找到相关日志条目")
            return self._empty_result()

        # 2. 计算隶属度最高的单条日志的加权隶属度
        metadata, sim_score = results[0]
        correctness = float(metadata.get("correctness_score", 0.0))
        retrieved_slices = (
            metadata.get("retrieved_slices", "").split("|")
            if metadata.get("retrieved_slices")
            else []
        )
        # 综合隶属度：mu = w1 * similarity + w2 * correctness
        membership = (w1 * sim_score) + (w2 * correctness)
        qualified = membership >= fit_threshold

        top_log = {
            "id": metadata.get("id", "unknown"),
            "question": metadata.get("question", ""),
            "similarity": float(sim_score),
            "correctness_score": correctness,
            "membership_degree": membership,
            "retrieved_slices": retrieved_slices,
            # 日志自带切片内容快照（命中后直接使用，无需再查切片库）
            "retrieved_slices_content": metadata.get("retrieved_slices_content", ""),
        }

        # 命中时，该条日志的切片即推荐切片
        weighted_slices = (
            [
                {
                    "slice_id": slice_id,
                    "membership_degree": membership,
                    "normalized_membership": 1.0,
                }
                for slice_id in retrieved_slices
                if slice_id
            ]
            if qualified
            else []
        )

        logger.info(
            "隶属度计算完成: 最大得分=%.4f (阈值=%.4f, 合格=%s), "
            "相关日志=1条, 推荐切片=%d个",
            membership,
            fit_threshold,
            qualified,
            len(weighted_slices),
        )

        return {
            "membership_score": membership,
            "max_membership": membership,
            "top_logs": [top_log],
            "weighted_slices": weighted_slices,
            "qualified_log_count": 1 if qualified else 0,
            "qualified_memberships": [membership] if qualified else [],
        }

    @staticmethod
    def _empty_result() -> dict:
        """返回空的隶属度结果"""
        return {
            "membership_score": 0.0,
            "top_logs": [],
            "weighted_slices": [],
            "qualified_log_count": 0,
            "qualified_memberships": [],
        }
