"""
临时脚本：从 SAR-VQA-180375 生成问答样本知识库语料（严格剔除 val 集，防测试泄漏）

- val 集 5000 条按 (image, question.lower()) 键全部剔除（180375 中命中 4436 条 id 重叠）
- 随机抽样 N 条控制向量库规模（默认 30000）
- 输出 agent/data/sar_vqa_samples.txt（builder 可识别的 txt 格式）
"""
import os
import sys

import k_means_constrained  # noqa: F401

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
for p in (current_dir, root_dir):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

VAL_CSV = os.path.join(root_dir, "dataset_split", "val.csv")
BIG_CSV = os.path.join(current_dir, "data", "SAR-VQA-180375.csv")
OUT_TXT = os.path.join(current_dir, "data", "sar_vqa_samples.txt")
SAMPLE_N = 30000
RANDOM_SEED = 42


def main():
    val = pd.read_csv(VAL_CSV)
    big = pd.read_csv(BIG_CSV)

    # val 键（image + question 小写，覆盖 180375 路径格式差异）
    val_img = val['image'].astype(str).str.replace(r"^/", "", regex=True)
    val_keys = set(zip(val_img.str.replace("\\", "/"), val['question'].astype(str).str.strip().str.lower()))

    def _key(row):
        img = str(row['image']).replace("\\", "/").lstrip("/")
        return (img, str(row['question']).strip().lower())

    keep = big[~big.apply(_key, axis=1).isin(val_keys)]
    dropped = len(big) - len(keep)
    print(f"原始 {len(big)} 条，剔除 val 泄漏 {dropped} 条，剩余 {len(keep)} 条")

    sampled = keep.sample(n=min(SAMPLE_N, len(keep)), random_state=RANDOM_SEED)
    print(f"抽样 {len(sampled)} 条")

    lines = []
    for _, r in sampled.iterrows():
        lines.append(f"Question: {str(r['question']).strip()}\nAnswer: {str(r['answer']).strip()}\n")
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"✅ 已写出: {OUT_TXT} ({os.path.getsize(OUT_TXT)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
