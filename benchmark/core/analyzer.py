"""
结果数据分析模块
包含置信度阈值计算、趋势可视化等功能
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ====================== 配置 ======================
DEFAULT_CONFIDENCE_THRESHOLD = 0.85
DEFAULT_TARGET_METRICS = ["cosine", "ROUGEL", "BLEU1", "BLEU2", "BLEU3", "BLEU4", "METEOR", "IG", "ID"]

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ====================== 置信度阈值计算 ======================
def calculate_conf_threshold(data: pd.DataFrame, col_name: str, confidence: float):
    """
    计算指定列的指定置信度阈值

    参数:
        data: 清洗后的数据集（需包含 'correct' 列）
        col_name: 列名
        confidence: 目标置信度

    返回:
        tuple: (满足置信度的最小阈值, 实际置信度, 样本数)
    """
    sorted_unique_vals = np.sort(data[col_name].unique())

    valid_thresholds = []
    valid_confidences = []

    for threshold in sorted_unique_vals:
        filtered_data = data[data[col_name] >= threshold]
        if len(filtered_data) == 0:
            continue

        current_confidence = filtered_data["correct"].mean()

        if current_confidence >= confidence:
            valid_thresholds.append(threshold)
            valid_confidences.append(current_confidence)

    if not valid_thresholds:
        return None, 0, 0

    best_threshold = min(valid_thresholds)
    best_confidence = valid_confidences[valid_thresholds.index(best_threshold)]
    best_sample_num = len(data[data[col_name] >= best_threshold])

    return best_threshold, best_confidence, best_sample_num


# ====================== 结果分析 ======================
class ResultAnalyzer:
    """结果数据分析器"""

    def __init__(
        self,
        csv_path: str,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        target_metrics: list = None
    ):
        """
        参数:
            csv_path: benchmark 结果 CSV 路径
            confidence_threshold: 置信度阈值
            target_metrics: 目标指标列表
        """
        self.csv_path = csv_path
        self.confidence_threshold = confidence_threshold
        self.target_metrics = target_metrics or DEFAULT_TARGET_METRICS
        self.df = None
        self.core_df = None
        self.metrics_avg = None
        self.results_summary = {}
        self.avg_pred_tokens = 0.0

    def load_data(self) -> bool:
        """加载并清洗数据"""
        try:
            self.df = pd.read_csv(self.csv_path, encoding="utf-8")

            self.df['pred_len'] = self.df['predicted'].str.len()
            self.avg_pred_tokens = self.df['pred_len'].mean()

            self.core_df = self.df[["correct"] + self.target_metrics].dropna()
            self.core_df["correct"] = self.core_df["correct"].astype(int)

            print(f"有效样本数量: {len(self.core_df)}")
            print(f"平均 Token 长度(字符数): {self.avg_pred_tokens:.2f}")
            print(f"数据中 correct=1 的样本占比: {self.core_df['correct'].mean():.4f}")

            self.metrics_avg = self.core_df[self.target_metrics].mean()
            return True

        except Exception as e:
            print(f"加载数据失败: {e}")
            return False

    def compute_thresholds(self) -> dict:
        """计算所有目标指标的置信度阈值"""
        self.results_summary = {}

        for metric in self.target_metrics:
            threshold, conf, sample = calculate_conf_threshold(
                self.core_df, metric, self.confidence_threshold
            )
            self.results_summary[metric] = {
                "threshold": threshold,
                "conf": conf,
                "sample": sample
            }

        return self.results_summary

    def print_summary(self):
        """打印分析摘要"""
        conf_percent = self.confidence_threshold * 100

        print("=" * 80)
        print("【各指标全量样本平均值统计】")
        print("-" * 80)
        for metric in self.target_metrics:
            print(f"  {metric:7} 平均值: {self.metrics_avg[metric]:.6f}")

        print("\n" + "=" * 80)
        print(f"【{conf_percent:.0f}% 置信度 各指标阈值计算结果】")
        print("-" * 80)

        for metric, res in self.results_summary.items():
            if res["threshold"] is not None:
                print(f"  {metric:7} 阈值: {res['threshold']:.6f} | 实际置信度: {res['conf']:.2%}"
                      f" | 样本数: {res['sample']}")
            else:
                print(f"  {metric:7} 未找到满足条件的阈值")
        print("=" * 80)

    def plot_confidence_trends(self, save_dir: str = None):
        """绘制所有指标的阈值-置信度趋势图"""
        if self.core_df is None:
            print("请先调用 load_data() 加载数据")
            return

        for metric in self.target_metrics:
            self._plot_single_trend(metric)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def _plot_single_trend(self, col_name: str):
        """绘制单个指标的置信度趋势图"""
        data = self.core_df
        sorted_vals = np.sort(data[col_name].unique())
        confidences = []

        for t in sorted_vals:
            filt = data[data[col_name] >= t]
            conf = filt["correct"].mean() if len(filt) > 0 else 0
            confidences.append(conf)

        plt.figure(figsize=(10, 5))
        plt.plot(sorted_vals, confidences, 'b-', linewidth=2, label=f'{col_name} - 置信度曲线')

        current_best_t = self.results_summary.get(col_name, {}).get("threshold")

        if current_best_t:
            plt.axvline(x=current_best_t, color='orange', linestyle=':', linewidth=2,
                        label=f'最优阈值: {current_best_t:.6f}')

        conf_percent = self.confidence_threshold * 100
        plt.xlabel(f'{col_name} 取值', fontsize=12)
        plt.ylabel('置信度 (correct=1 的概率)', fontsize=12)
        plt.title(f'{col_name} 取值与 correct=1 的置信度关系（目标 {conf_percent:.0f}% 置信度）', fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    def analyze(self, plot: bool = False, save_dir: str = None) -> dict:
        """
        执行完整分析流程

        参数:
            plot: 是否绘制趋势图
            save_dir: 图表保存目录

        返回:
            dict: 分析结果摘要
        """
        if not self.load_data():
            return {}

        self.compute_thresholds()
        self.print_summary()

        if plot:
            self.plot_confidence_trends(save_dir)

        return self.results_summary
