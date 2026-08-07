"""
LLM 语义判断模块
通过 model/factory.py 的 LangChain 模型调用，替代原 rag_rsfit_builder.py 中的原生 requests.post
实现 LLMJudgeClient 抽象基类
"""
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from rag.base.llm_judge_client import LLMJudgeClient
from model.factory import doubao_1_5_lite_model
from utils.logger_handler import logger


# 判断提示词模板
LLM_SYSTEM_PROMPT = "你是一个语义匹配判断助手，仅返回1或0，不解释原因"
LLM_PROMPT_TEMPLATE = """
Question: {question}
Ground Truth Answer: {answer}
Predicted Answer: {predicted}
Does the predicted answer match the ground truth? Answer 1 for match and 0 for not match.
Use semantic meaning not exact match. Synonyms are also treated as a match, e.g., football and soccer,
playground and ground track field, building and rooftop, pond and swimming pool. Do not explain the reason.
""".strip()

LLM_MAX_WORKERS = 50  # 并发线程数


class LlmJudge(LLMJudgeClient):
    """
    LLM 语义判断实现
    使用 factory.py 中的 doubao_1_5_lite_model 通过 LangChain 调用
    """

    def __init__(self, model=None, max_workers: int = LLM_MAX_WORKERS,
                 max_retries: int = 3, retry_delay: float = 1.0):
        """
        Args:
            model: LangChain 聊天模型实例，默认使用 doubao_1_5_lite_model
            max_workers: 并发线程数
            max_retries: 单次判断失败后的重试次数（指数退避）
            retry_delay: 首次重试等待秒数，之后每次翻倍
        """
        self._model = model or doubao_1_5_lite_model
        self._max_workers = max_workers
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    def judge(self, question: str, ground_truth: str, predicted: str) -> str:
        """
        判断预测答案是否与标准答案语义匹配

        Args:
            question: 问题文本
            ground_truth: 标准答案
            predicted: 预测答案

        Returns:
            '1'（匹配）或 '0'（不匹配）
        """
        prompt = LLM_PROMPT_TEMPLATE.format(
            question=question, answer=ground_truth, predicted=predicted
        )

        # 瞬时失败（500/限流/网络抖动）重试：指数退避，全部重试失败才返回 "0"
        import time

        for attempt in range(max(1, self._max_retries)):
            try:
                response = self._model.invoke([
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ])

                # 提取文本响应
                if hasattr(response, "content"):
                    content = response.content.strip()
                elif isinstance(response, str):
                    content = response.strip()
                else:
                    content = str(response).strip()

                if content not in ["0", "1"]:
                    content = "1" if "1" in content else ("0" if "0" in content else "0")

                return content

            except Exception as e:
                if attempt < max(1, self._max_retries) - 1:
                    delay = self._retry_delay * (2 ** attempt)
                    logger.warning(
                        f"[LlmJudge] 判断调用失败，{delay:.1f}秒后重试 "
                        f"({attempt + 1}/{self._max_retries}): {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.warning(
                        f"[LlmJudge] 判断失败（重试 {self._max_retries} 次仍失败）: {e}"
                    )
                    return "0"

        return "0"

    def judge_and_filter(
        self,
        test_df,
        new_logs: list,
        max_workers: int = None,
    ) -> tuple:
        """
        对新日志进行 LLM 判断，过滤出正确和错误的内容（并行版本）

        Args:
            test_df: 原始测试数据 DataFrame（包含 answer 列）
            new_logs: 新产生的日志列表
            max_workers: 最大并发线程数

        Returns:
            tuple: (correct_logs, error_logs)
        """
        if not new_logs:
            return [], []

        import pandas as pd

        max_workers = min(max_workers or self._max_workers, len(new_logs))
        answer_map = {}
        for _, row in test_df.iterrows():
            answer_map[str(row["id"])] = (
                str(row["answer"]) if pd.notna(row["answer"]) else ""
            )

        correct_logs = []
        error_logs = []
        total = len(new_logs)
        start_time = None
        import time

        start_time = time.time()
        logger.info(
            f"[LLM判断] 开始对 {total} 条新日志进行大模型判断，并发数: {max_workers}..."
        )

        def process_single_log(log_entry):
            try:
                entry_id = str(log_entry["id"])
                question = log_entry["question"]
                predicted = log_entry["predicted_text"]
                answer = answer_map.get(entry_id, "")

                llm_result = self.judge(question, answer, predicted)

                if llm_result == "1":
                    return {"type": "correct", "entry": log_entry}
                else:
                    error_entry = log_entry.copy()
                    error_entry["llm_judge"] = "0"
                    error_entry["ground_truth"] = answer
                    return {"type": "error", "entry": error_entry}
            except Exception as e:
                logger.error(f"[LLM判断] 处理日志ID {log_entry.get('id')} 失败: {e}")
                return {"type": "error", "entry": log_entry, "error": str(e)}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_log = {
                executor.submit(process_single_log, log): log for log in new_logs
            }

            for completed_count, future in enumerate(
                as_completed(future_to_log), start=1
            ):
                try:
                    result = future.result()
                    if result["type"] == "correct":
                        correct_logs.append(result["entry"])
                    else:
                        error_logs.append(result["entry"])
                except Exception as e:
                    error_logs.append(future_to_log[future])

                if completed_count % 10 == 0 or completed_count == total:
                    elapsed = time.time() - start_time
                    speed = completed_count / elapsed if elapsed > 0 else 0.0
                    progress = completed_count / total * 100 if total else 100.0

                    logger.info(
                        f"[LLM判断进度] {completed_count}/{total} ({progress:.1f}%) | "
                        f"速度: {speed:.2f} 条/秒 | "
                        f"正确: {len(correct_logs)} | 错误: {len(error_logs)}"
                    )

        # 保存错误日志
        if error_logs:
            self._save_error_logs(error_logs)
        else:
            logger.info(f"[LLM判断] 无错误日志，无需保存")

        logger.info(
            f"[LLM判断] 完成！正确: {len(correct_logs)} | 错误: {len(error_logs)} | "
            f"正确率: {len(correct_logs) / total * 100 if total > 0 else 0:.2f}%"
        )

        return correct_logs, error_logs

    def rejudge_existing_logs(self, test_df, log_df) -> tuple:
        """
        一次性数据迁移：对现有日志进行 LLM 重新判断

        Args:
            test_df: 原始测试数据 DataFrame
            log_df: 现有日志 DataFrame

        Returns:
            tuple: (correct_logs, error_logs)
        """
        import pandas as pd

        if log_df.empty:
            logger.warning("[一次性重判] 当前日志为空，无需重判")
            return [], []

        logger.info(f"[一次性重判] 开始对现有 {len(log_df)} 条日志进行LLM判断...")

        existing_logs = log_df.to_dict("records")
        for log in existing_logs:
            for key in log:
                if pd.isna(log[key]):
                    log[key] = ""

        correct_logs, error_logs = self.judge_and_filter(test_df, existing_logs)

        if not correct_logs:
            logger.warning("[一次性重判] 所有日志均被判定为错误，保留原数据")
            return correct_logs, error_logs

        logger.info(f"[一次性重判] 用 {len(correct_logs)} 条正确日志重建日志文件")
        return correct_logs, error_logs

    def _save_error_logs(self, error_logs: list):
        """保存错误日志到 CSV 文件"""
        import os
        import pandas as pd

        from rag.core.config import rag_config

        if not error_logs:
            return

        error_df = pd.DataFrame(error_logs)
        cols = [
            "id",
            "question",
            "retrieved_slices",
            "correctness_score",
            "predicted_text",
            "ground_truth",
            "llm_judge",
            "timestamp",
        ]
        for col in cols:
            if col not in error_df.columns:
                error_df[col] = ""
        error_df = error_df[cols]

        if os.path.exists(rag_config.error_log_path):
            existing_df = pd.read_csv(rag_config.error_log_path)
            error_df = pd.concat([existing_df, error_df], ignore_index=True)
            error_df = error_df.drop_duplicates(subset=["id"], keep="last")

        os.makedirs(os.path.dirname(rag_config.error_log_path), exist_ok=True)
        error_df.to_csv(rag_config.error_log_path, index=False, encoding="utf-8")
        logger.info(
            f"[LLM判断] 错误日志已保存至: {rag_config.error_log_path}，共 {len(error_logs)} 条"
        )
