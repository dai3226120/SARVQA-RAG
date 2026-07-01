import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.path_tool import get_abs_path

# 配置文件路径（和脚本放在同一文件夹）
# input_path = get_abs_path("SAR-VQA_ALL-200103_filtered_question-re50+.csv")
# output_path = get_abs_path("SAR-VQA_ALL-180375_filtered_dataset-SAR-ship_answer-empty.csv")
# input_path = get_abs_path("dataset_analysis/origin_dataset/Landsat30-AU-VQA-train_utf8.csv")
# output_path = get_abs_path("dataset_analysis/origin_dataset/Landsat30-AU-VQA-train_answer-empty.csv")
# input_path = get_abs_path("dataset_analysis/origin_dataset/2024EarthVQA_QA_utf8.csv")
# output_path = get_abs_path("dataset_analysis/origin_dataset/2024EarthVQA_QA_answer-empty.csv")
input_path = get_abs_path("dataset_analysis/origin_dataset/SARLANG-1M_all_utf8.csv")
output_path = get_abs_path("dataset_analysis/origin_dataset/SARLANG-1M_all_answer-empty.csv")

# 读取原始CSV
df = pd.read_csv(input_path, encoding="utf-8-sig")

# 核心过滤逻辑
# 保留规则：image不包含指定路径 + answer非空（排除空值/空字符串/全空格）
filter_rule = (
    # ~df["image"].str.contains(r"/SAR-TEXT-data/Image/SAR-ship", na=False)
    # & df["answer"].notna()
    # & (df["answer"].str.strip() != "")
    df["answer"].notna()
    & (df["answer"].str.strip() != "")
)
df_filtered = df[filter_rule]

# 保存结果，不生成额外索引列
df_filtered.to_csv(output_path, index=False, encoding="utf-8-sig")

# 控制台输出处理统计
print(f"原始总行数：{len(df)}")
print(f"过滤后剩余行数：{len(df_filtered)}")
print(f"已删除行数：{len(df) - len(df_filtered)}")
print(f"处理完成，结果已保存至：{output_path}")