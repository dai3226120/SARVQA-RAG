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
        k: int = None,
        fit_threshold: float = None,
        top_p: int = None,
        w1: float = None,
        w2: float = None,
    ) -> dict:
        """
        计算新问题与日志中相关内容的隶属度

        Args:
            query: 新查询问题
            k: 检索相关日志条目的数量
            fit_threshold: 隶属度阈值
            top_p: 最多返回的合格隶属度数量
            w1: 相似度权重（运行时透传，优先于构造默认）
            w2: 正确性分数权重（运行时透传，优先于构造默认）

        Returns:
            dict: 包含 membership_score / max_membership / top_logs /
                  weighted_slices / qualified_log_count / qualified_memberships
        """
        k = k or rag_config.membership_k
        fit_threshold = fit_threshold if fit_threshold is not None else rag_config.fit_threshold
        top_p = top_p or rag_config.top_p

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

        # 1. 在日志库中全量精确检索相关条目
        try:
            results = self._index.search_text(query, k)
        except Exception as e:
            logger.error("日志库全量精确检索失败: %s", e)
            return self._empty_result()

        if not results:
            logger.warning("未找到相关日志条目")
            return self._empty_result()

        # 2. 计算加权隶属度
        top_logs = []
        slice_memberships = {}
        total_membership = 0.0
        qualified_memberships = []
        all_memberships = []

        for metadata, sim_score in results:
            correctness = float(metadata.get("correctness_score", 0.0))
            retrieved_slices = (
                metadata.get("retrieved_slices", "").split("|")
                if metadata.get("retrieved_slices")
                else []
            )

            # 综合隶属度：mu = w1 * similarity + w2 * correctness
            membership = (w1 * sim_score) + (w2 * correctness)
            total_membership += membership
            all_memberships.append(membership)

            top_logs.append({
                "id": metadata.get("id", "unknown"),
                "question": metadata.get("question", ""),
                "similarity": float(sim_score),
                "correctness_score": correctness,
                "membership_degree": membership,
                "retrieved_slices": retrieved_slices,
            })

            # 仅基于合格日志统计切片隶属度（取最高隶属度）
            if membership >= fit_threshold:
                qualified_memberships.append(membership)
                for slice_id in retrieved_slices:
                    if slice_id:
                        if slice_id not in slice_memberships or membership > slice_memberships[slice_id]:
                            slice_memberships[slice_id] = membership

        avg_membership = total_membership / len(top_logs) if top_logs else 0.0
        max_membership = max(all_memberships) if all_memberships else 0.0

        # 对合格隶属度按降序排序，最多保留 top_p 个
        qualified_memberships.sort(reverse=True)
        qualified_memberships = qualified_memberships[:top_p]

        # 生成加权切片列表
        weighted_slices = [
            {
                "slice_id": slice_id,
                "membership_degree": membership,
                "normalized_membership": membership / max_membership
                if max_membership > 0
                else 0.0,
            }
            for slice_id, membership in sorted(
                slice_memberships.items(), key=lambda x: x[1], reverse=True
            )
        ]

        logger.info(
            "隶属度计算完成: 平均得分=%.4f, 最大得分=%.4f, "
            "相关日志=%d条, 合格日志=%d条, 推荐切片=%d个",
            avg_membership,
            max_membership,
            len(top_logs),
            len(qualified_memberships),
            len(weighted_slices),
        )

        return {
            "membership_score": avg_membership,
            "max_membership": max_membership,
            "top_logs": top_logs,
            "weighted_slices": weighted_slices,
            "qualified_log_count": len(qualified_memberships),
            "qualified_memberships": qualified_memberships,
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
