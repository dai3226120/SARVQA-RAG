"""
语义缓存系统协调器
源自原 rag_rsfit_builder.py SARSemanticCacheSystem 的协调部分：
  - evaluate_and_log（并行评估+记录）
  - calculate_membership_degree（委托给 MembershipCalculator）
  - __init__（委托给 LogManager + VlmEvaluator + LlmJudge）
"""
import os
from datetime import datetime

import pandas as pd

from rag.membership.log_manager import LogManager, fetch_slices_content
from rag.membership.degree_calculator import MembershipCalculator
from rag.membership.vlm_evaluator import VlmEvaluator
from rag.membership.llm_judge import LlmJudge
from rag.stores.slice_store import SliceStore
from rag.core.config import rag_config
from rag.core.exact_index import ExactVectorIndex
from utils.logger_handler import logger


class SemanticCacheSystem:
    """
    SAR RAG 测试与隶属度计算系统（协调器）

    核心流程：
    1. evaluate_and_log: 对测试数据进行 RAG 检索 + 多模态模型预测，评估正确性并记录日志
    2. calculate_membership_degree: 新问题与日志做隶属度计算，获得隶属度得分和推荐切片
    """

    def __init__(
        self,
        force_full_reload: bool = False,
        clear_vector_db_only: bool = False,
    ):
        """
        Args:
            force_full_reload: 强制全量重载
            clear_vector_db_only: 只清除向量库
        """
        # 委托给各子模块
        self._log_manager = LogManager(
            force_full_reload=force_full_reload,
            clear_vector_db_only=clear_vector_db_only,
        )
        self._vlm_evaluator = VlmEvaluator()
        self._llm_judge = LlmJudge()
        # 日志库全量精确检索器（numpy 暴力 top-k，替代 Chroma HNSW 近似）
        self._logs_index = ExactVectorIndex(
            collection=self._log_manager.logs_collection._collection,
            persist_directory=rag_config.persist_directory,
            collection_name=rag_config.log_collection_name,
            embedding_fn=self._log_manager._embeddings,
        )
        self._calculator = MembershipCalculator(self._logs_index)

    @property
    def log_df(self) -> pd.DataFrame:
        return self._log_manager.log_df

    @log_df.setter
    def log_df(self, value: pd.DataFrame):
        self._log_manager.log_df = value

    def calculate_membership_degree(
        self, query: str, k: int = None, fit_threshold: float = None,
        top_p: int = None, w1: float = None, w2: float = None,
    ) -> dict:
        """
        计算新问题与日志中相关内容的隶属度
        委托给 MembershipCalculator

        Args:
            query: 新查询问题
            k: 检索相关日志条目的数量
            fit_threshold: 隶属度阈值
            top_p: 最多返回的合格隶属度数量
            w1: 相似度权重（运行时透传，优先于构造默认）
            w2: 正确性分数权重（运行时透传，优先于构造默认）

        Returns:
            dict: 隶属度计算结果
        """
        return self._calculator.calculate(
            query, k=k, fit_threshold=fit_threshold, top_p=top_p, w1=w1, w2=w2
        )

    def evaluate_and_log(self, test_df, base_rag_engine=None, max_workers=50):
        """
        步骤1：执行测试并记录 CSV 日志（并行版本）

        Args:
            test_df: 测试数据 DataFrame
            base_rag_engine: RAG 引擎实例（兼容旧接口，当前不再使用）
            max_workers: 最大并发线程数
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time

        # 获取已存在的 ID 集合
        existing_ids = set()
        if not self.log_df.empty and "id" in self.log_df.columns:
            existing_ids = set(self.log_df["id"].dropna().astype(str).tolist())

        test_rows = []
        for _, row in test_df.iterrows():
            current_id = str(row["id"])
            if current_id not in existing_ids:
                test_rows.append(row)
            else:
                logger.info(
                    "[增量跳过] 测试ID: %s 已经存在于日志中，跳过本次评估。", current_id
                )

        if not test_rows:
            logger.info("[增量检查完成] 所有测试数据均已存在于日志中，未触发新测试。")
            return

        def process_single_row(row):
            try:
                logger.info(
                    "[测试ID: %s] 问题: %s", row["id"], row["question"]
                )

                # image 列为相对路径（如 /SAR-TEXT-data/...），拼接 base_image_dir 得绝对路径
                image_full = os.path.join(
                    rag_config.base_image_dir, str(row["image"]).lstrip("/")
                )
                score, response_text, slice_ids = self._vlm_evaluator.call_vlm_agent(
                    image_full, row["question"], row.get("answer")
                )

                logger.info(
                    "[测试ID: %s] 预测结果: %s", row["id"], response_text
                )
                logger.info(
                    "[测试ID: %s] 正确性得分: %.4f", row["id"], score
                )
                logger.info(
                    "[测试ID: %s] agent实际使用的切片ID: %s",
                    row["id"],
                    slice_ids,
                )

                return {
                    "id": row["id"],
                    "question": row["question"],
                    "retrieved_slices": "|".join(slice_ids),
                    "correctness_score": score,
                    "predicted_text": response_text,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            except Exception as e:
                logger.error(
                    "[测试ID: %s] 处理失败: %s", row["id"], str(e), exc_info=True
                )
                return None

        new_logs = []
        total_tasks = len(test_rows)
        start_time = time.time()
        failed_count = 0

        logger.info(
            "[并行处理] 开始测试，共 %d 条任务，并发数: %d", total_tasks, max_workers
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_row = {
                executor.submit(process_single_row, row): row for row in test_rows
            }

            for completed_count, future in enumerate(
                as_completed(future_to_row), start=1
            ):
                try:
                    log_entry = future.result()
                    if log_entry is not None:
                        new_logs.append(log_entry)
                    else:
                        failed_count += 1
                except Exception as e:
                    failed_count += 1
                    row = future_to_row[future]
                    logger.error("[ERROR] 测试ID: %s 执行出错: %s", row["id"], str(e))

                if completed_count % 10 == 0 or completed_count == total_tasks:
                    elapsed = time.time() - start_time
                    speed = completed_count / elapsed if elapsed > 0 else 0.0
                    progress = completed_count / total_tasks * 100 if total_tasks else 100.0
                    remaining = (
                        (total_tasks - completed_count) / speed if speed > 0 else 0
                    )

                    logger.info(
                        "[PROGRESS] 进度更新 [%d/%d] (%.1f%%) 速度: %.2f 条/秒 "
                        "已完成: %d条有效记录 | 失败: %d条",
                        completed_count,
                        total_tasks,
                        progress,
                        speed,
                        len(new_logs),
                        failed_count,
                    )

        # LLM 判断过滤
        if new_logs:
            correct_logs, error_logs = self._llm_judge.judge_and_filter(
                test_df, new_logs
            )

            if correct_logs:
                # 补新 schema 字段：correct（LLM 判断通过恒为 1）+ 切片内容
                # （综合向量 = 问题+切片内容+回答 在线路径的组成部分）
                slice_store = SliceStore()
                for log in correct_logs:
                    log["correct"] = 1
                    slice_ids = str(log.get("retrieved_slices", "")).split("|")
                    log["retrieved_slices_content"] = fetch_slices_content(
                        slice_store, slice_ids
                    )

                new_df = pd.DataFrame(correct_logs)
                self.log_df = pd.concat([self.log_df, new_df], ignore_index=True)
                self._log_manager._deduplicate_log_df()

                # 增量写入向量库
                for log_entry in correct_logs:
                    self._log_manager.add_single_record(log_entry)

                self._log_manager.update_md5()

                logger.info(
                    f"日志已更新（LLM判断正确），新增 {len(correct_logs)} 条记录，"
                    f"当前总条数: {len(self.log_df)}"
                )
            else:
                logger.warning(
                    f"[LLM判断] 所有 {len(new_logs)} 条新日志均被判定为错误"
                )

            logger.info(
                f"[LLM判断汇总] 正确: {len(correct_logs)} | 错误: {len(error_logs)}"
            )
        else:
            logger.info("[增量检查完成] 所有测试数据均已存在于日志中，未触发新测试。")

    def rejudge_existing_logs(self, test_df):
        """
        对现有日志进行 LLM 重新判断

        Args:
            test_df: 原始测试数据 DataFrame
        """
        correct_logs, error_logs = self._llm_judge.rejudge_existing_logs(
            test_df, self.log_df
        )

        if not correct_logs:
            logger.warning("[一次性重判] 所有日志均被判定为错误，保留原数据")
            return

        self.log_df = pd.DataFrame(correct_logs)
        cols = [
            "id",
            "question",
            "retrieved_slices",
            "correctness_score",
            "predicted_text",
            "timestamp",
        ]
        for col in cols:
            if col not in self.log_df.columns:
                self.log_df[col] = ""
        self.log_df = self.log_df[cols]

        self._log_manager._save_log_df()
        self._log_manager.reload_to_vector_db()
        self._log_manager.update_md5()

        logger.info(
            f"[一次性重判] 完成！正确: {len(correct_logs)} | 错误: {len(error_logs)}"
        )


# 兼容旧导入：保留 Config 类引用
class Config:
    """兼容旧代码的配置类（建议改用 rag.core.config.rag_config）"""
    retrieval_k = rag_config.k
    membership_k = rag_config.membership_k
    slice_k = rag_config.slice_k
    top_p = rag_config.top_p
    membership_w1 = rag_config.w1
    membership_w2 = rag_config.w2
    bleu_weight = rag_config.bleu_weight
    overlap_weight = rag_config.overlap_weight
    test_nrows = rag_config.test_nrows
    base_image_dir = rag_config.base_image_dir
    log_name = rag_config.log_name
    error_log_name = rag_config.error_log_name
    collection_name = rag_config.log_collection_name
    vlm_agent_type = rag_config.vlm_agent_type
