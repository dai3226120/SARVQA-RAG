"""
Excel 统计表模块
生成/增量更新评估结果汇总表
"""

import os
import pandas as pd

from config import path_config


# ====================== Excel 统计表配置 ======================
SUMMARY_EXCEL_PATH = os.path.join(path_config.RESULT_DIR, "eval_summary.xlsx")
SUMMARY_COLUMNS = [
    "运行时间戳", "模型类型", "文件标签", "输出目录",
    # Agent 调用统计
    "Agent-总调用次数", "Agent-成功次数", "Agent-成功率(%)", "Agent-总耗时(秒)", "Agent-平均耗时(秒)",
    # 检索工具统计
    "检索-调用次数", "检索-总耗时(秒)", "检索-平均耗时(秒)",
    # 隶属度命中率
    "隶属度命中率(%)",
    # 结果评估统计
    "评估-总行数", "评估-成功行数", "评估-成功率(%)",
    "评估-语义匹配率(%)", "评估-平均IG", "评估-平均ID",
    # 各指标全量样本平均值
    "指标-cosine", "指标-ROUGEL", "指标-BLEU1", "指标-BLEU2", "指标-BLEU3", "指标-BLEU4", "指标-METEOR",
    # 85% 置信度统计（样本数）
    "置信度-cosine样本数", "置信度-ROUGEL样本数", "置信度-BLEU1样本数", "置信度-BLEU2样本数",
    "置信度-BLEU3样本数", "置信度-BLEU4样本数", "置信度-METEOR样本数",
]


def _r(val, ndigits=4):
    """安全 round"""
    try:
        return round(float(val), ndigits)
    except (TypeError, ValueError):
        return 0


def update_excel_summary(timestamp: str, model_type: str, file_tag: str,
                         output_dir: str, pred_result: dict, bench_result: dict,
                         analysis_result: dict):
    """增量更新 Excel 统计表"""
    pred_stats = pred_result.get("stats", {})
    retrieval_stats = pred_result.get("retrieval_stats", {})
    bench_stats = bench_result.get("stats", {})
    analysis_stats = analysis_result.get("stats", {})
    metrics_avg = analysis_result.get("metrics_avg", {})
    results_summary = analysis_result.get("results_summary", {})

    new_row = {
        "运行时间戳": timestamp,
        "模型类型": model_type,
        "文件标签": file_tag,
        "输出目录": output_dir,
        # Agent 调用统计
        "Agent-总调用次数": pred_stats.get("call_count", 0),
        "Agent-成功次数": pred_stats.get("success_count", 0),
        "Agent-成功率(%)": _r(pred_stats.get("success_rate", 0), 2),
        "Agent-总耗时(秒)": _r(pred_stats.get("total_latency", 0), 2),
        "Agent-平均耗时(秒)": _r(pred_stats.get("avg_latency", 0), 4),
        # 检索工具统计
        "检索-调用次数": retrieval_stats.get("call_count", 0),
        "检索-总耗时(秒)": _r(retrieval_stats.get("total_latency", 0), 4),
        "检索-平均耗时(秒)": _r(retrieval_stats.get("avg_latency", 0), 4),
        # 隶属度命中率
        "隶属度命中率(%)": _r(pred_result.get("membership_hit_rate", 0) * 100, 2)
        if pred_result.get("membership_hit_rate") is not None else "N/A",
        # 结果评估统计
        "评估-总行数": bench_stats.get("total_rows", 0),
        "评估-成功行数": bench_stats.get("success_count", 0),
        "评估-成功率(%)": _r(
            (bench_stats.get("success_count", 0) / bench_stats.get("total_rows", 1) * 100)
            if bench_stats.get("total_rows", 0) > 0 else 0, 2
        ),
        "评估-语义匹配率(%)": _r(bench_stats.get("match_rate", 0) * 100, 2),
        "评估-平均IG": _r(bench_stats.get("avg_ig", 0), 4),
        "评估-平均ID": _r(bench_stats.get("avg_id", 0), 4),
        # 各指标全量样本平均值
        "指标-cosine": _r(metrics_avg.get("cosine", 0), 6),
        "指标-ROUGEL": _r(metrics_avg.get("ROUGEL", 0), 6),
        "指标-BLEU1": _r(metrics_avg.get("BLEU1", 0), 6),
        "指标-BLEU2": _r(metrics_avg.get("BLEU2", 0), 6),
        "指标-BLEU3": _r(metrics_avg.get("BLEU3", 0), 6),
        "指标-BLEU4": _r(metrics_avg.get("BLEU4", 0), 6),
        "指标-METEOR": _r(metrics_avg.get("METEOR", 0), 6),
        # 85% 置信度统计（样本数）
        "置信度-cosine样本数": results_summary.get("cosine", {}).get("sample", 0),
        "置信度-ROUGEL样本数": results_summary.get("ROUGEL", {}).get("sample", 0),
        "置信度-BLEU1样本数": results_summary.get("BLEU1", {}).get("sample", 0),
        "置信度-BLEU2样本数": results_summary.get("BLEU2", {}).get("sample", 0),
        "置信度-BLEU3样本数": results_summary.get("BLEU3", {}).get("sample", 0),
        "置信度-BLEU4样本数": results_summary.get("BLEU4", {}).get("sample", 0),
        "置信度-METEOR样本数": results_summary.get("METEOR", {}).get("sample", 0),
    }

    new_df = pd.DataFrame([new_row], columns=SUMMARY_COLUMNS)

    # 读取已有 Excel（如果存在），追加新行
    if os.path.exists(SUMMARY_EXCEL_PATH):
        try:
            existing_df = pd.read_excel(SUMMARY_EXCEL_PATH)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        except Exception:
            combined_df = new_df
    else:
        combined_df = new_df

    os.makedirs(os.path.dirname(SUMMARY_EXCEL_PATH), exist_ok=True)
    try:
        combined_df.to_excel(SUMMARY_EXCEL_PATH, index=False, engine='openpyxl')
        actual_path = SUMMARY_EXCEL_PATH
    except PermissionError:
        # 文件被占用时（如 Excel 正在打开），使用备用文件名
        fallback_path = SUMMARY_EXCEL_PATH.replace(".xlsx", f"_{timestamp}.xlsx")
        combined_df.to_excel(fallback_path, index=False, engine='openpyxl')
        actual_path = fallback_path
        print(f"\n⚠️ 主统计表被占用，已保存到备用文件")
    print(f"\n📊 统计表已更新: {actual_path}")
    print(f"   累计记录数: {len(combined_df)}")
