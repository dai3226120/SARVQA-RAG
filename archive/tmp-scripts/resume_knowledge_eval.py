"""
临时脚本：knowledge 全量评测续跑（按 id 跳过已完成的样本）

全量评测进程多次在长时间运行后被外部终止，已保存的 2700 行结果有效（0 错误）。
本脚本：
1. 读已保存预测 CSV + val.csv，按 id 找出缺失样本
2. 对缺失样本用与 main_eval.py 相同的调用路径（registry + 重试配置）重新预测
3. 按 id 合并写回两个预测 CSV（保留原列序）
"""
import k_means_constrained  # noqa: F401

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

import config as cfg
from core import process_vqa_data, create_process_row_func
from main_eval import _MODEL_REGISTRY, API_MAX_RETRIES, API_RETRY_DELAY, IMAGE_BASE_PATH, MAX_WORKERS

RESULT_DIR = os.path.join(_PROJECT_ROOT, "benchmark/result/agent-doubao/agent-text-doubao-seed-2-0-mini_knowledge/20260812_072641")
LATEST_CSV = os.path.join(RESULT_DIR, "agent-text-doubao-seed-2-0-mini_knowledge_val_predicted_question_latest.csv")
ORIG_CSV = os.path.join(RESULT_DIR, "agent-text-doubao-seed-2-0-mini_knowledge_val_predicted_question.csv")
VAL_CSV = cfg.get_csv_input_path("val")
MODEL_KEY = "agent-text-doubao-seed_knowledge"


def main():
    saved = pd.read_csv(LATEST_CSV, encoding='utf-8-sig')
    val = pd.read_csv(VAL_CSV, encoding='utf-8-sig')

    # 合并已知结果：主 CSV + 上次续跑未合并的临时结果（增量续跑）
    known = saved.copy()
    tmp_latest = os.path.join(os.path.join(RESULT_DIR, "resume_tmp"), "resumed_predicted_latest.csv")
    if os.path.exists(tmp_latest):
        tmp_df = pd.read_csv(tmp_latest, encoding='utf-8-sig')
        known = pd.concat([known, tmp_df[['id', 'image', 'question', 'answer', 'predicted', 'IG', 'ID']]],
                          ignore_index=True)
        known = known.drop_duplicates(subset='id', keep='first')
        print(f"含上次续跑未合并结果 {len(tmp_df)} 行，已知结果合计 {len(known)} 行")

    print(f"主 CSV 已保存 {len(saved)} 行 | val 共 {len(val)} 行")

    known_ids = set(known['id'])
    missing = val[~val['id'].isin(known_ids)]
    print(f"缺失样本: {len(missing)} 行")

    if missing.empty:
        print("无缺失样本，无需续跑")
        return

    # 1. 临时输入 CSV
    tmp_dir = os.path.join(RESULT_DIR, "resume_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_csv = os.path.join(tmp_dir, "missing_input.csv")
    missing[['id', 'image', 'question', 'answer']].to_csv(tmp_csv, index=False, encoding='utf-8-sig')

    # 2. 用与 main_eval.py 相同的调用路径重预测
    entry = _MODEL_REGISTRY[MODEL_KEY]
    process_func = create_process_row_func(
        entry["api_call"],
        include_metrics=True,
        prompt_template=cfg.prompt_config.DEFAULT_PROMPT,
        max_retries=API_MAX_RETRIES,
        retry_delay=API_RETRY_DELAY,
    )
    result_df = process_vqa_data(
        process_func=process_func,
        csv_input_path=tmp_csv,
        result_dir=tmp_dir,
        base_filename="resumed_predicted",
        image_base_path=IMAGE_BASE_PATH,
        use_timestamp=False,
        max_rows=10000,
        start_row=0,
        max_workers=MAX_WORKERS,
        batch_save_threshold=100,
    )
    if result_df is None or len(result_df) == 0:
        print("❌ 续跑无结果")
        sys.exit(1)

    # 3. 合并写回（主 CSV + 本次新增）
    new_df = pd.read_csv(os.path.join(tmp_dir, "resumed_predicted_latest.csv"), encoding='utf-8-sig')
    repl_map = {row['id']: row for _, row in new_df.iterrows()}

    merged = val.copy()
    # 统一 lower 化与 main_eval 输出一致（question/answer/predicted）
    for col in ['question', 'answer']:
        merged[col] = merged[col].astype(str).str.lower()
    merged['predicted'] = ""
    merged['IG'] = 0.0
    merged['ID'] = 0.0

    matched = 0
    for idx, rid in enumerate(merged['id']):
        if rid in repl_map:
            r = repl_map[rid]
            merged.at[idx, 'predicted'] = r['predicted']
            merged.at[idx, 'IG'] = r.get('IG', 0)
            merged.at[idx, 'ID'] = r.get('ID', 0)
            matched += 1
        elif rid in known_ids:
            s = known[known['id'] == rid].iloc[0]
            merged.at[idx, 'predicted'] = s['predicted']
            merged.at[idx, 'IG'] = s.get('IG', 0)
            merged.at[idx, 'ID'] = s.get('ID', 0)
            matched += 1
    print(f"合并完成: {matched}/{len(merged)} 行")

    merged.to_csv(LATEST_CSV, index=False, encoding='utf-8-sig')
    merged.to_csv(ORIG_CSV, index=False, encoding='utf-8-sig')
    print(f"✅ 已写回: {LATEST_CSV} / {ORIG_CSV}")

    err = merged['predicted'].astype(str).str.lower().str.startswith('error').sum()
    print(f"最终错误行: {err}")


if __name__ == "__main__":
    main()
