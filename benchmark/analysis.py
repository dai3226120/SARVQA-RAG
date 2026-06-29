"""
分析模块
包含置信度阈值计算和可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt

from config import analysis_config


def calculate_conf_threshold(data, col_name, confidence=None):
    """
    计算指定列的指定置信度阈值

    参数:
        data: 数据集（DataFrame）
        col_name: 列名
        confidence: 目标置信度（默认使用配置值）

    返回:
        tuple: (最优阈值, 实际置信度, 样本数)
    """
    if confidence is None:
        confidence = analysis_config.CONFIDENCE_THRESHOLD

    # 对目标列的值升序排序+去重
    sorted_unique_vals = np.sort(data[col_name].unique())

    valid_thresholds = []
    valid_confidences = []

    for threshold in sorted_unique_vals:
        filtered_data = data[data[col_name] >= threshold]
        if len(filtered_data) == 0:
            continue

        # 计算该阈值下的置信度
        current_confidence = filtered_data["correct"].mean()

        if current_confidence >= confidence:
            valid_thresholds.append(threshold)
            valid_confidences.append(current_confidence)

    if not valid_thresholds:
        return None, 0, 0

    # 取最小的阈值作为最优阈值
    best_threshold = min(valid_thresholds)
    best_confidence = valid_confidences[valid_thresholds.index(best_threshold)]
    best_sample_num = len(data[data[col_name] >= best_threshold])

    return best_threshold, best_confidence, best_sample_num


def batch_calculate_thresholds(data, metrics=None):
    """
    批量计算所有指标的阈值

    参数:
        data: 数据集（DataFrame）
        metrics: 指标列名列表（默认使用配置值）

    返回:
        dict: 各指标的阈值计算结果
    """
    if metrics is None:
        metrics = analysis_config.TARGET_METRICS

    results_summary = {}

    for metric in metrics:
        threshold, conf, sample = calculate_conf_threshold(data, metric)
        results_summary[metric] = {
            "threshold": threshold,
            "conf": conf,
            "sample": sample
        }

    return results_summary


def print_analysis_results(data, results_summary, metrics=None):
    """
    打印分析结果

    参数:
        data: 数据集（DataFrame）
        results_summary: 阈值计算结果
        metrics: 指标列名列表
    """
    if metrics is None:
        metrics = analysis_config.TARGET_METRICS

    # 设置中文字体防止乱码
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 计算指标平均值
    metrics_avg = data[metrics].mean()

    conf_percent = analysis_config.CONFIDENCE_THRESHOLD * 100

    # 打印各指标平均值
    print("=" * 80)
    print(f"【各指标全量样本平均值统计】")
    print("-" * 80)
    for metric in metrics:
        print(f"📊 {metric:7} 平均值: {metrics_avg[metric]:.6f}")

    # 打印阈值计算结果
    print("\n" + "=" * 80)
    print(f"【{conf_percent:.0f}%置信度 各指标阈值计算结果】")
    print("-" * 80)

    for metric, res in results_summary.items():
        if res["threshold"] is not None:
            print(f"✅ {metric:7} 阈值: {res['threshold']:.6f} | 实际置信度: {res['conf']:.2%}"
                  f" | 样本数: {res['sample']}")
        else:
            print(f"❌ {metric:7} 未找到满足条件的阈值")
    print("=" * 80)

    return metrics_avg


def plot_confidence_trend(data, col_name, results_summary):
    """
    绘制阈值-置信度趋势图

    参数:
        data: 数据集（DataFrame）
        col_name: 列名
        results_summary: 阈值计算结果
    """
    sorted_vals = np.sort(data[col_name].unique())
    confidences = []

    for t in sorted_vals:
        filt = data[data[col_name] >= t]
        conf = filt["correct"].mean() if len(filt) > 0 else 0
        confidences.append(conf)

    plt.figure(figsize=(10, 5))
    plt.plot(sorted_vals, confidences, 'b-', linewidth=2, label=f'{col_name} - 置信度曲线')

    current_best_t = results_summary[col_name]["threshold"]
    conf_percent = analysis_config.CONFIDENCE_THRESHOLD * 100

    if current_best_t:
        plt.axvline(x=current_best_t, color='orange', linestyle=':', linewidth=2,
                    label=f'最优阈值: {current_best_t:.6f}')

    plt.xlabel(f'{col_name} 取值', fontsize=12)
    plt.ylabel('置信度 (correct=1的概率)', fontsize=12)
    plt.title(f'{col_name} 取值与correct=1的置信度关系（目标{conf_percent:.0f}%置信度）', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_all_confidence_trends(data, results_summary, metrics=None):
    """
    绘制所有指标的置信度趋势图

    参数:
        data: 数据集（DataFrame）
        results_summary: 阈值计算结果
        metrics: 指标列名列表
    """
    if metrics is None:
        metrics = analysis_config.TARGET_METRICS

    for metric in metrics:
        plot_confidence_trend(data, metric, results_summary)
