import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# 添加项目根目录到Python路径（根据实际项目结构调整）
sys.path.append(os.path.abspath("."))

# 导入rag_rsfit_builder中的核心类
from rag.rag_rsfit_builder import SARSemanticCacheSystem, Config

# ==============================================
# 1. 配置参数与初始化系统
# ==============================================
# 目标查询问题
TARGET_QUESTION = "Are there any agricultural fields or cultivated land visible near the buildings?"

# 日志文件路径（替换为实际路径）
LOG_FILE_PATH = "./agent/data/rag_feedback_logs.csv"

# 初始化隶属度计算系统
system = SARSemanticCacheSystem()
system.log_path = LOG_FILE_PATH  # 强制指定日志文件路径
if os.path.exists(system.log_path):
    system.log_df = pd.read_csv(system.log_path)
    # 确保日志数据格式正确
    system._deduplicate_log_df()
    system._check_and_reload_vector_db()
else:
    raise FileNotFoundError(f"日志文件不存在：{system.log_path}")

# ==============================================
# 2. 调用隶属度计算方法
# ==============================================
membership_result = system.calculate_membership_degree(
    query=TARGET_QUESTION,
    k=Config.membership_k,
    w1=Config.membership_w1,
    w2=Config.membership_w2
)

# 提取top_logs数据用于绘图
top_logs = membership_result["top_logs"]
if not top_logs:
    raise ValueError("未检索到相关日志条目，无法生成散点图")

# 转换为DataFrame方便处理
logs_df = pd.DataFrame(top_logs)
logs_df["id"] = logs_df["id"].astype(str)
logs_df["similarity"] = logs_df["similarity"].astype(float)
logs_df["correctness_score"] = logs_df["correctness_score"].astype(float)
logs_df["membership_degree"] = logs_df["membership_degree"].astype(float)

# 找出隶属度最大值、最小值行索引
max_mu_idx = logs_df["membership_degree"].idxmax()
min_mu_idx = logs_df["membership_degree"].idxmin()

# ==============================================
# 3. 绘制四象限散点图
# ==============================================
# 设置中文字体（可选，根据需要调整）
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 创建画布
fig, ax = plt.subplots(figsize=(12, 8))

# 计算象限分割线（以均值为界）
x_mean = logs_df["similarity"].mean()
y_mean = logs_df["correctness_score"].mean()

# 绘制象限分割线
ax.axvline(x=x_mean, color="gray", linestyle="--", alpha=0.7, label=f"相似度均值: {x_mean:.4f}")
ax.axhline(y=y_mean, color="gray", linestyle="--", alpha=0.7, label=f"正确性均值: {y_mean:.4f}")

# 定义象限颜色和标签
quadrant_colors = {
    "高相似-高正确": "#2E8B57",  # 深绿
    "高相似-低正确": "#FF6347",  # 番茄红
    "低相似-高正确": "#4682B4",  # 钢蓝
    "低相似-低正确": "#DAA520"   # 金铜色
}

# 分类每个点到对应象限
def get_quadrant(x, y):
    if x >= x_mean and y >= y_mean:
        return "高相似-高正确"
    elif x >= x_mean and y < y_mean:
        return "高相似-低正确"
    elif x < x_mean and y >= y_mean:
        return "低相似-高正确"
    else:
        return "低相似-低正确"

logs_df["quadrant"] = logs_df.apply(lambda row: get_quadrant(row["similarity"], row["correctness_score"]), axis=1)

# 绘制每个象限的散点：原s=100 → 缩小至1/3 ≈ 33
for quadrant, color in quadrant_colors.items():
    quad_data = logs_df[logs_df["quadrant"] == quadrant]
    ax.scatter(
        quad_data["similarity"],
        quad_data["correctness_score"],
        c=color,
        label=quadrant,
        s=33,  # 尺寸改为原来1/3（原100 → 33）
        alpha=0.8,
        edgecolors="black",
        linewidth=0.5
    )

# ========== 仅标注隶属度最大、最小两个点，其余点不打标签 ==========
# 最大值标注
max_row = logs_df.loc[max_mu_idx]
ax.annotate(
    f"MAX\n{max_row['id']}\nμ:{max_row['membership_degree']:.4f}",
    xy=(max_row["similarity"], max_row["correctness_score"]),
    xytext=(8, 8),
    textcoords="offset points",
    fontsize=7,
    bbox=dict(boxstyle="round,pad=0.3", fc="#90EE90", ec="darkgreen", alpha=0.8)
)

# 最小值标注
min_row = logs_df.loc[min_mu_idx]
ax.annotate(
    f"MIN\n{min_row['id']}\nμ:{min_row['membership_degree']:.4f}",
    xy=(min_row["similarity"], min_row["correctness_score"]),
    xytext=(8, 8),
    textcoords="offset points",
    fontsize=7,
    bbox=dict(boxstyle="round,pad=0.3", fc="#FFC0CB", ec="darkred", alpha=0.8)
)

# 设置图表属性
ax.set_title(f"问题隶属度四象限分析\n目标问题: {TARGET_QUESTION}", fontsize=14, pad=20)
ax.set_xlabel("相似度 (Similarity)", fontsize=12)
ax.set_ylabel("正确性分数 (Correctness Score)", fontsize=12)
ax.legend(loc="upper right", fontsize=10)
ax.grid(True, alpha=0.3)

# 象限背景浅色
ax.add_patch(Rectangle((x_mean, y_mean), 1-x_mean, 1-y_mean, color=quadrant_colors["高相似-高正确"], alpha=0.1))
ax.add_patch(Rectangle((x_mean, 0), 1-x_mean, y_mean, color=quadrant_colors["高相似-低正确"], alpha=0.1))
ax.add_patch(Rectangle((0, y_mean), x_mean, 1-y_mean, color=quadrant_colors["低相似-高正确"], alpha=0.1))
ax.add_patch(Rectangle((0, 0), x_mean, y_mean, color=quadrant_colors["低相似-低正确"], alpha=0.1))

# 调整布局并保存
plt.tight_layout()
plt.savefig("./membership_quadrant_scatter.png", dpi=300, bbox_inches="tight")
plt.show()

# ==============================================
# 4. 输出统计信息
# ==============================================
print("="*80)
print("隶属度计算结果统计:")
print(f"目标问题: {TARGET_QUESTION}")
print(f"综合隶属度得分: {membership_result['membership_score']:.4f}")
print(f"检索到的相关日志数: {len(top_logs)}")
print(f"最大隶属度条目 ID: {max_row['id']} | μ={max_row['membership_degree']:.4f}")
print(f"最小隶属度条目 ID: {min_row['id']} | μ={min_row['membership_degree']:.4f}")
print("\n各象限数据统计:")
for quadrant in quadrant_colors.keys():
    quad_data = logs_df[logs_df["quadrant"] == quadrant]
    count = len(quad_data)
    avg_membership = quad_data["membership_degree"].mean() if count > 0 else 0.0
    print(f"  {quadrant}: {count} 条记录, 平均隶属度: {avg_membership:.4f}")
print("="*80)

# 输出详细数据
print("\n详细日志数据:")
print(logs_df[["id", "similarity", "correctness_score", "membership_degree", "quadrant"]].to_string(index=False))