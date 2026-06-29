import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------- 1. 配置参数与加载数据 --------------------------
# FILE_TAG = "doubao-seed-2-0-mini-260428"
# FILE_TAG = "internvl2-8b"
# FILE_TAG = "internvl3_5-8b"

# FILE_TAG = "agent-text-doubao-seed-2-0-mini"
FILE_TAG = "agent-text-doubao-seed-2-0-mini_rscsv"
# FILE_TAG = "agent-text-internVL2-8b"
# FILE_TAG = "agent-text-internVL3_5-8b"

# DATASET_TAG = "test"
DATASET_TAG = "val"
CSV_INPUT_PATH = f"./dataset_split/{DATASET_TAG}.csv"
IMAGE_BASE_PATH = "C:\\dataset/SAR-TEXT"  # 图像文件基础路径
RESULT_DIR = "./benchmark/result/benchmark_result/"
BASE_FILENAME = f"{FILE_TAG}_{DATASET_TAG}_benchmark"
file_path = f"{RESULT_DIR}/{BASE_FILENAME}_latest.csv"
# file_path = f"{RESULT_DIR}/{BASE_FILENAME}_20260624_234239.csv"
# 置信度要求（可动态配置）
CONFIDENCE_THRESHOLD = 0.85
# 设置中文字体防止乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 更新需要处理的列名列表
target_metrics = ["cosine", "ROUGEL", "BLEU1", "BLEU2", "BLEU3", "BLEU4", "METEOR", "IG", "ID"]
# target_metrics = ["cosine", "ROUGEL"]

# 修改加载逻辑：动态包含所有指标列
df = pd.read_csv(file_path, encoding="utf-8")

# 计算predicted文本token平均长度
df['pred_len'] = df['predicted'].str.len()
avg_pred_tokens = df['pred_len'].mean()

core_df = df[["correct"] + target_metrics].dropna()
core_df["correct"] = core_df["correct"].astype(int)

print(f"有效样本数量: {len(core_df)}")
print(f"平均Token长度(字符数): {avg_pred_tokens:.2f}")
print(f"数据中correct=1的样本占比: {core_df['correct'].mean():.4f}")
# 计算 target_metrics 中每一列的平均值
metrics_avg = core_df[target_metrics].mean()

# -------------------------- 2. 定义通用阈值计算函数 --------------------------
def calculate_conf_threshold(data: pd.DataFrame, col_name: str, confidence: float):
    """
    计算指定列(余弦/ROUGEL)的指定置信度阈值
    :param data: 清洗后的数据集
    :param col_name: 列名，只能是 "cosine" 或 "ROUGEL"
    :param confidence: 目标置信度，由CONFIDENCE_THRESHOLD配置
    :return: 满足置信度的最小阈值、该阈值下的实际置信度、该阈值下的样本数
    """
    # 对目标列的值 升序排序+去重，生成所有候选阈值
    sorted_unique_vals = np.sort(data[col_name].unique())
    
    valid_thresholds = []  # 存储满足置信度的所有阈值
    valid_confidences = [] # 存储对应阈值的实际置信度

    for threshold in sorted_unique_vals:
        # 筛选：当前列值 >= 候选阈值的所有样本
        filtered_data = data[data[col_name] >= threshold]
        if len(filtered_data) == 0:
            continue
        
        # 计算该阈值下的置信度 = correct=1的样本数 / 筛选出的总样本数
        current_confidence = filtered_data["correct"].mean()
        
        # 筛选满足置信度要求的阈值
        if current_confidence >= confidence:
            valid_thresholds.append(threshold)
            valid_confidences.append(current_confidence)

    if not valid_thresholds:
        return None, 0, 0
    # 取最小的阈值作为最优阈值（满足条件的前提下，阈值越小越好）
    best_threshold = min(valid_thresholds)
    # 找到最优阈值对应的实际置信度
    best_confidence = valid_confidences[valid_thresholds.index(best_threshold)]
    # 最优阈值下的样本量
    best_sample_num = len(data[data[col_name] >= best_threshold])
    
    return best_threshold, best_confidence, best_sample_num

# -------------------------- 3. 批量计算所有指标的阈值 --------------------------
results_summary = {}

for metric in target_metrics:
    threshold, conf, sample = calculate_conf_threshold(core_df, metric, CONFIDENCE_THRESHOLD)
    results_summary[metric] = {
        "threshold": threshold,
        "conf": conf,
        "sample": sample
    }

# -------------------------- 4. 输出结果 --------------------------
conf_percent = CONFIDENCE_THRESHOLD * 100
print("=" * 80)
print(f"【各指标全量样本平均值统计】")
print("-" * 80)
for metric in target_metrics:
    # 打印每个指标的算术平均值
    print(f"📊 {metric:7} 平均值: {metrics_avg[metric]:.6f}")

print("\n" + "=" * 80)
print(f"【{conf_percent:.0f}%置信度 各指标阈值计算结果】")
print("-" * 80)

for metric, res in results_summary.items():
    if res["threshold"] is not None:
        # 展示满足置信度要求的最小阈值、实际置信度和样本量
        print(f"✅ {metric:7} 阈值: {res['threshold']:.6f} | 实际置信度: {res['conf']:.2%}" 
              f" | 样本数: {res['sample']}")
    else:
        print(f"❌ {metric:7} 未找到满足条件的阈值")
print("=" * 80)

# -------------------------- 5. 可视化：阈值-置信度趋势图（可选，辅助分析） --------------------------
def plot_confidence_trend(data, col_name):
    sorted_vals = np.sort(data[col_name].unique())
    confidences = []
    for t in sorted_vals:
        filt = data[data[col_name] >= t]
        conf = filt["correct"].mean() if len(filt) > 0 else 0
        confidences.append(conf)
    
    plt.figure(figsize=(10, 5))
    plt.plot(sorted_vals, confidences, 'b-', linewidth=2, label=f'{col_name} - 置信度曲线')
    # 动态获取当前指标计算出的阈值
    current_best_t = results_summary[col_name]["threshold"]
    
    if current_best_t:
        plt.axvline(x=current_best_t, color='orange', linestyle=':', linewidth=2, 
                    label=f'最优阈值: {current_best_t:.6f}')
    
    plt.xlabel(f'{col_name} 取值', fontsize=12)
    plt.ylabel('置信度 (correct=1的概率)', fontsize=12)
    plt.title(f'{col_name} 取值与correct=1的置信度关系（目标{conf_percent:.0f}%置信度）', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# 遍历绘制所有指标的趋势图
for metric in target_metrics:
    plot_confidence_trend(core_df, metric)