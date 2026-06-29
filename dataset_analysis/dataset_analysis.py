import pandas as pd
import re
from collections import Counter

# ---------------------- 1. 读取数据 ----------------------
# 请确保csv文件路径正确
# df = pd.read_csv("./dataset_analysis/origin_dataset/SAR-VQA_ALL-180375_filtered_dataset-SAR-ship_answer-empty.csv")
df = pd.read_csv("./dataset_analysis/origin_dataset/Landsat30-AU-VQA-train_answer-empty.csv")
# df = pd.read_csv("./dataset_analysis/origin_dataset/2024EarthVQA_QA_answer-empty.csv")
# df = pd.read_csv("./dataset_analysis/origin_dataset/SARLANG-1M_all_answer-empty.csv")

# ---------------------- 2. 提取数据集名称 ----------------------
# 从 image 字段提取数据集名称：/SAR-TEXT-data/Image/XXX/
def extract_dataset_name(image_path):
    # match = re.search(r'/SAR-TEXT-data/Image/([^/]+)/', str(image_path))
    # # match = re.search(r'DEA_VLM_images/([^/]+)/', str(image_path))
    # return match.group(1) if match else None


    # 针对你的2024EarthVQA数据集，image字段只有文件名，没有路径
    # 所以所有数据都属于同一个数据集"2024EarthVQA"
    # return "2024EarthVQA"
    # return "SARLANG-1M_all"
    return "Landsat30-AU-VQA-train"

# df['dataset'] = df['image'].apply(extract_dataset_name)
df['dataset'] = df['image_path'].apply(extract_dataset_name)

# 过滤掉dataset为NaN的行（防止意外情况）
df = df.dropna(subset=['dataset'])

# ---------------------- 3. 维度1：按数据集统计 ----------------------
dataset_stats = []
dataset_detail_list = []

for dataset_name, group in df.groupby('dataset'):
    # 基础统计
    total_count = len(group)
    # 问题/答案平均字数
    avg_q_len = group['question'].str.len().mean()
    avg_a_len = group['answer'].str.len().mean()
    # 不重复问题数量
    unique_questions = group['question'].unique()
    unique_q_count = len(unique_questions)
    # 每个问题出现次数
    q_counter = Counter(group['question'])
    
    # 汇总统计
    dataset_stats.append({
        "数据集名称": dataset_name,
        "总数据条数": total_count,
        "不重复问题数": unique_q_count,
        "问题平均字数": round(avg_q_len, 2),
        "答案平均字数": round(avg_a_len, 2)
    })
    
    # 每个问题的出现次数
    for q, cnt in q_counter.items():
        dataset_detail_list.append({
            "数据集名称": dataset_name,
            "问题": q,
            "出现次数": cnt
        })

# 转DataFrame
dataset_summary_df = pd.DataFrame(dataset_stats)
dataset_detail_df = pd.DataFrame(dataset_detail_list)

# ---------------------- 4. 维度2：按问题统计（出现的数据集） ----------------------
question_to_datasets = {}
question_count = {}
for idx, row in df.iterrows():
    q = row['question']
    ds = row['dataset']
    if q not in question_to_datasets:
        question_to_datasets[q] = set()
        question_count[q] = 0
    question_to_datasets[q].add(ds)
    question_count[q] += 1

question_stats = []
for q, datasets in question_to_datasets.items():
    question_stats.append({
        "问题": q,
        "出现总次数": question_count[q],
        "出现的数据集数量": len(datasets),
        "出现的数据集列表": ", ".join(sorted(datasets))
    })
question_df = pd.DataFrame(question_stats)

# ---------------------- 5. 输出CSV文件 ----------------------
output_dir = "./dataset_analysis/dataset_analysis_result/"

# dataset_summary_df.to_csv(output_dir + "数据集总统计_SAR-VQA_ALL-180375_filtered_dataset-SAR-ship_answer-empty.csv", index=False, encoding="utf-8-sig")
# dataset_detail_df.to_csv(output_dir + "数据集-问题出现次数_SAR-VQA_ALL-180375_filtered_dataset-SAR-ship_answer-empty.csv", index=False, encoding="utf-8-sig")
# question_df.to_csv(output_dir + "问题-出现的数据集_SAR-VQA_ALL-180375_filtered_dataset-SAR-ship_answer-empty.csv", index=False, encoding="utf-8-sig")

dataset_summary_df.to_csv(output_dir + "数据集总统计_Landsat30-AU-VQA-train_answer-empty.csv", index=False, encoding="utf-8-sig")
dataset_detail_df.to_csv(output_dir + "数据集-问题出现次数_Landsat30-AU-VQA-train_answer-empty.csv", index=False, encoding="utf-8-sig")
question_df.to_csv(output_dir + "问题-出现的数据集_Landsat30-AU-VQA-train_answer-empty.csv", index=False, encoding="utf-8-sig")

# dataset_summary_df.to_csv(output_dir + "数据集总统计_2024EarthVQA_QA_answer-empty.csv", index=False, encoding="utf-8-sig")
# dataset_detail_df.to_csv(output_dir + "数据集-问题出现次数_2024EarthVQA_QA_answer-empty.csv", index=False, encoding="utf-8-sig")
# question_df.to_csv(output_dir + "问题-出现的数据集_2024EarthVQA_QA_answer-empty.csv", index=False, encoding="utf-8-sig")

# dataset_summary_df.to_csv(output_dir + "数据集总统计_SARLANG-1M_all_answer-empty.csv", index=False, encoding="utf-8-sig")
# dataset_detail_df.to_csv(output_dir + "数据集-问题出现次数_SARLANG-1M_all_answer-empty.csv", index=False, encoding="utf-8-sig")
# question_df.to_csv(output_dir + "问题-出现的数据集_SARLANG-1M_all_answer-empty.csv", index=False, encoding="utf-8-sig")

# ---------------------- 6. 生成Markdown报告 ----------------------
md_content = "# SAR-VQA 数据统计分析报告\n\n"

# 总览
md_content += "## 一、数据总览\n"
md_content += f"- 总数据条数：{len(df)}\n"
md_content += f"- 总数据集数量：{df['dataset'].nunique()}\n"
md_content += f"- 总不重复问题数：{df['question'].nunique()}\n\n"

# 数据集统计
md_content += "## 二、各数据集统计详情\n"
for _, row in dataset_summary_df.iterrows():
    md_content += f"### {row['数据集名称']}\n"
    md_content += f"- 数据条数：{row['总数据条数']}\n"
    md_content += f"- 不重复问题数：{row['不重复问题数']}\n"
    md_content += f"- 问题平均字数：{row['问题平均字数']}\n"
    md_content += f"- 答案平均字数：{row['答案平均字数']}\n\n"

# 问题跨数据集统计
md_content += "## 三、问题跨数据集出现情况\n"
md_content += f"共 {len(question_df)} 个不重复问题，以下是每个问题出现的数据集：\n\n"
for _, row in question_df.iterrows():
    md_content += f"- **问题**：{row['问题']}\n"
    md_content += f"  - 出现数据集数量：{row['出现的数据集数量']}\n"
    md_content += f"  - 数据集列表：{row['出现的数据集列表']}\n\n"

# 保存MD
# with open(output_dir + "SAR-VQA数据统计报告_SAR-VQA_ALL-180375_filtered_dataset-SAR-ship_answer-empty.md", "w", encoding="utf-8") as f:
#     f.write(md_content)
with open(output_dir + "SAR-VQA数据统计报告_Landsat30-AU-VQA-train_answer-empty.md", "w", encoding="utf-8") as f:
    f.write(md_content)
# with open(output_dir + "SAR-VQA数据统计报告_2024EarthVQA_QA_answer-empty.md", "w", encoding="utf-8") as f:
#     f.write(md_content)
# with open(output_dir + "SAR-VQA数据统计报告_SARLANG-1M_all_answer-empty.md", "w", encoding="utf-8") as f:
#     f.write(md_content)

print("✅ 统计完成！生成文件如下：")
print("1. 数据集总统计.csv")
print("2. 数据集-问题出现次数.csv")
print("3. 问题-出现的数据集.csv")
print("4. SAR-VQA数据统计报告.md")