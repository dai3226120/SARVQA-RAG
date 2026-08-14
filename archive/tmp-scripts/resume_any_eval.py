"""
通用增量续跑脚本：按 id 跳过已完成样本，续跑缺失部分。
用法: python benchmark/resume_any_eval.py <result_dir> <timestamp> <model_key>
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

RESULT_ROOT = os.path.join(_PROJECT_ROOT, "benchmark/result")
VAL_CSV = cfg.get_csv_input_path("val")


def main():
    if len(sys.argv) < 4:
        print("用法: resume_any_eval.py <模型类型目录> <时间戳> <model_key>")
        sys.exit(1)
    model_type_dir, timestamp, model_key = sys.argv[1], sys.argv[2], sys.argv[3]
    result_dir = os.path.join(RESULT_ROOT, model_type_dir, timestamp)
    entry = _MODEL_REGISTRY[model_key]
    tag = entry["file_tag"]
    latest_csv = os.path.join(result_dir, f"{tag}_val_predicted_question_latest.csv")
    orig_csv = os.path.join(result_dir, f"{tag}_val_predicted_question.csv")

    saved = pd.read_csv(latest_csv, encoding='utf-8-sig')
    val = pd.read_csv(VAL_CSV, encoding='utf-8-sig')

    # 启动时先把上次续跑的临时结果合并固化进主 CSV（防止被 kill 后进度回退）
    tmp_latest = os.path.join(result_dir, "resume_tmp", "resumed_predicted_latest.csv")
    if os.path.exists(tmp_latest):
        tmp_df = pd.read_csv(tmp_latest, encoding='utf-8-sig')
        merged_known = pd.concat(
            [saved[['id', 'image', 'question', 'answer', 'predicted', 'IG', 'ID']],
             tmp_df[['id', 'image', 'question', 'answer', 'predicted', 'IG', 'ID']]],
            ignore_index=True).drop_duplicates(subset='id', keep='first')
        merged_known.to_csv(latest_csv, index=False, encoding='utf-8-sig')
        merged_known.to_csv(orig_csv, index=False, encoding='utf-8-sig')
        print(f"已将上次临时结果 {len(tmp_df)} 行固化到主 CSV，主 CSV 现有 {len(merged_known)} 行")
        saved = merged_known

    known = saved.copy()
    print(f"主 CSV {len(saved)} 行 | val {len(val)} 行")

    known_ids = set(known['id'])
    missing = val[~val['id'].isin(known_ids)]
    print(f"缺失样本: {len(missing)} 行")
    if missing.empty:
        print("无缺失，无需续跑")
        return

    tmp_dir = os.path.join(result_dir, "resume_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_csv = os.path.join(tmp_dir, "missing_input.csv")
    missing[['id', 'image', 'question', 'answer']].to_csv(tmp_csv, index=False, encoding='utf-8-sig')

    # Agent 模型用 AGENT_PROMPT + agent_client（与 main_eval.py full 模式一致，
    # 保证续跑部分与首次预测的提示词/IG-ID 计算方式相同）
    is_agent = entry.get("is_agent", False)
    process_func = create_process_row_func(
        entry["api_call"], include_metrics=True,
        prompt_template=cfg.prompt_config.AGENT_PROMPT if is_agent else cfg.prompt_config.DEFAULT_PROMPT,
        agent_client=entry.get("client") if is_agent else None,
        max_retries=API_MAX_RETRIES, retry_delay=API_RETRY_DELAY,
    )
    result_df = process_vqa_data(
        process_func=process_func,
        csv_input_path=tmp_csv,
        result_dir=tmp_dir,
        base_filename="resumed_predicted",
        image_base_path=IMAGE_BASE_PATH,
        use_timestamp=False,
        max_rows=10000, start_row=0,
        max_workers=MAX_WORKERS, batch_save_threshold=100,
    )
    if result_df is None or len(result_df) == 0:
        print("❌ 续跑无结果")
        sys.exit(1)

    new_df = pd.read_csv(os.path.join(tmp_dir, "resumed_predicted_latest.csv"), encoding='utf-8-sig')
    repl_map = {row['id']: row for _, row in new_df.iterrows()}

    merged = val.copy()
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
    merged.to_csv(latest_csv, index=False, encoding='utf-8-sig')
    merged.to_csv(orig_csv, index=False, encoding='utf-8-sig')
    err = merged['predicted'].astype(str).str.lower().str.startswith('error').sum()
    print(f"✅ 已写回 | 最终错误行: {err}")


if __name__ == "__main__":
    main()
