"""
测试入口脚本
源自原 rag_rsfit_builder.py run_real_test()

运行模式通过 RUN_MODE 变量控制：
  - "incremental": 增量更新模式（默认）
  - "force_reload": 强制全量重载模式
  - "rejudge": 重判模式
"""
import os
import pandas as pd

from rag.membership.cache_system import SemanticCacheSystem
from rag.core.config import rag_config
from utils.logger_handler import logger


# 运行模式配置
RUN_MODE = "incremental"


def run_real_test():
    """
    运行真实数据的测试
    """
    logger.info(f"=== 开始 SemanticCacheSystem 真实数据测试 [模式: {RUN_MODE}] ===")

    force_full_reload = RUN_MODE == "force_reload"
    clear_vector_db_only = RUN_MODE == "rejudge"

    system = SemanticCacheSystem(
        force_full_reload=force_full_reload,
        clear_vector_db_only=clear_vector_db_only,
    )

    # rejudge 模式：清除向量库后直接入库
    if RUN_MODE == "rejudge":
        logger.info("--- [重判模式] 清除向量库并重新入库 ---")
        if not system.log_df.empty:
            system._log_manager.reload_to_vector_db()
            logger.info(
                f"[重判模式] 成功入库 {len(system.log_df)} 条日志记录"
            )
        else:
            logger.warning("[重判模式] 日志为空，无需入库")
        logger.info("=== 重判模式完成 ===")
        return

    # 创建 RAG 引擎
    from rag.eval.real_rag_engine import RealRAGEngine

    base_rag_engine = RealRAGEngine()

    # 读取测试数据
    test_csv_path = _find_test_csv()
    if not test_csv_path or not os.path.exists(test_csv_path):
        logger.error(f"找不到测试文件: {test_csv_path}")
        return

    test_df = pd.read_csv(test_csv_path, nrows=rag_config.test_nrows)

    logger.info(f"测试数据加载成功，文件路径: {test_csv_path}")
    logger.info(f"加载了 {len(test_df)} 条测试数据")

    # 修正图片路径
    test_df["image"] = test_df["image"].apply(
        lambda x: os.path.join(rag_config.base_image_dir, x.lstrip("/"))
    )

    # 步骤1: 评估和记录
    logger.info("--- 步骤1: 执行评估和记录 ---")
    try:
        system.evaluate_and_log(test_df, base_rag_engine)
    except Exception as e:
        logger.exception("步骤1出错: %s", e)
        return

    logger.info(f"当前日志条数: {len(system.log_df)}")

    # 步骤2: 隶属度计算测试
    logger.info("--- 步骤2: 隶属度计算测试 ---")
    if len(system.log_df) > 0:
        try:
            query = test_df.iloc[0]["question"]
            membership_result = system.calculate_membership_degree(query)

            logger.info("隶属度计算结果:")
            logger.info(
                "  综合隶属度得分: %.4f",
                membership_result["membership_score"],
            )
            logger.info(
                "  相关日志数: %d", len(membership_result["top_logs"])
            )

            if membership_result["top_logs"]:
                for log in membership_result["top_logs"][:2]:
                    logger.info(
                        "    ID=%s, 相似度=%.4f, 正确性=%.4f, 隶属度=%.4f",
                        log["id"],
                        log["similarity"],
                        log["correctness_score"],
                        log["membership_degree"],
                    )
        except Exception as e:
            logger.exception("步骤2出错: %s", e)

    logger.info("=== 测试完成 ===")


def _find_test_csv() -> str:
    """查找测试 CSV 文件路径"""
    candidates = [
        "../../dataset_split/test.csv",
        "../dataset_split/test.csv",
        "dataset_split/test.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


if __name__ == "__main__":
    run_real_test()
