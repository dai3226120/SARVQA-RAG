"""
SAR-VQA结果评估脚本 - 执行入口
"""

import os
import sys
import csv
import shutil
import time
import numpy as np
import threading
import concurrent.futures

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

import pandas as pd
import nltk

from config import doubao_config, prompt_config, analysis_config
from utils import lock
from metrics import calculate_cosine_similarity, calculate_rouge_l, calculate_all_metrics


def call_chat_api(question, answer, predicted):
    """调用豆包API判断语义匹配"""
    import requests

    formatted_prompt = prompt_config.AGENT_PROMPT.format(
        question=question,
        answer=answer,
        predicted=predicted
    )

    data = {
        "model": doubao_config.MODEL_NAME,
        "temperature": float(doubao_config.TEMPERATURE),
        "messages": [
            {"role": "system", "content": "你是一个语义匹配判断助手，仅返回1或0，不解释原因"},
            {"role": "user", "content": formatted_prompt}
        ],
        "thinking": {"type": "disabled"}
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {doubao_config.API_KEY}"
    }

    try:
        response = requests.post(
            doubao_config.API_ENDPOINT,
            json=data,
            headers=headers,
            timeout=doubao_config.TIMEOUT
        )

        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                message_content = result['choices'][0].get('message', {}).get('content', '').strip()
                if message_content not in ["0", "1"]:
                    message_content = "1" if "1" in message_content else "0" if "0" in message_content else "0"
                return message_content
            return "0"
        else:
            with lock:
                print(f"API调用失败: {response.status_code}")
            return "0"

    except Exception as e:
        with lock:
            print(f"API请求异常: {e}")
        return "0"


def process_single_row(row_data):
    """处理单行数据：计算相似度 + 调用API判断匹配度"""
    try:
        row_num, row = row_data

        id_str = str(row['id']) if pd.notna(row['id']) else ""
        image_path = str(row['image']) if pd.notna(row['image']) else ""
        question = str(row['question']) if pd.notna(row['question']) else ""
        answer = str(row['answer']) if pd.notna(row['answer']) else ""
        predicted = str(row['predicted']) if pd.notna(row['predicted']) else ""

        with lock:
            print(f"  处理行 {row_num} | ID：{id_str} | 问题：{question[:50]}...")

        # 计算相似度指标
        cosine_sim = calculate_cosine_similarity(predicted, answer)
        rouge_l = calculate_rouge_l(predicted, answer)

        # 调用API判断匹配度
        correct = call_chat_api(question, answer, predicted)

        # 计算高级指标
        adv_metrics = calculate_all_metrics(predicted, answer)

        result = {
            "id": id_str,
            "image": image_path,
            "question": question,
            "answer": answer,
            "predicted": predicted,
            "correct": correct,
            "cosine": round(cosine_sim, 4),
            "ROUGEL": round(rouge_l, 4),
            "BLEU1": adv_metrics['BLEU1'],
            "BLEU2": adv_metrics['BLEU2'],
            "BLEU3": adv_metrics['BLEU3'],
            "BLEU4": adv_metrics['BLEU4'],
            "METEOR": adv_metrics['METEOR'],
            "status": "success"
        }

        with lock:
            print(f"  行 {row_num} 完成 | 匹配: {correct} | 余弦: {cosine_sim:.4f} | ROUGE-L: {rouge_l:.4f}")
        return result

    except Exception as e:
        error_msg = f" 行 {row_num} 失败：{str(e)}"
        with lock:
            print(error_msg)
        return {
            "id": "", "image": "", "question": "", "answer": "", "predicted": "",
            "correct": "0", "cosine": 0.0, "ROUGEL": 0.0,
            "BLEU1": 0.0, "BLEU2": 0.0, "BLEU3": 0.0, "BLEU4": 0.0, "METEOR": 0.0,
            "status": "failed", "error": error_msg
        }


def main():
    """主函数"""
    print("[START] 开始并行处理SAR-VQA结果评估...")

    # 配置参数
    # FILE_TAG = "agent-text-internVL2-8b"
    FILE_TAG = "agent-text-doubao-seed-2-0-mini"
    # FILE_TAG = "doubao-seed-2-0-mini-260428"
    # DATASET_TAG = "test"
    DATASET_TAG = "val"
    INPUT_CSV_PATH = f'./benchmark/result/{FILE_TAG}/{FILE_TAG}_{DATASET_TAG}_predicted_question_latest.csv'
    # INPUT_CSV_PATH = f'./benchmark/result/{FILE_TAG}/agent-text-doubao-seed-2-0-mini_val_predicted_question_20260624_223756.csv'
    RESULT_DIR = "./benchmark/result/benchmark_result/"
    BASE_FILENAME = f"{FILE_TAG}_{DATASET_TAG}_benchmark"
    MAX_ROWS = 50000
    MAX_WORKERS = 100

    OUTPUT_FIELD_NAMES = [
        'id', 'image', 'question', 'answer', 'predicted',
        'correct', 'cosine', 'ROUGEL', 'BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'METEOR'
    ]

    # 前置检查
    if not os.path.exists(os.path.dirname(INPUT_CSV_PATH)):
        print(f"❌ 输入目录不存在: {os.path.dirname(INPUT_CSV_PATH)}")
        return
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"❌ 输入文件不存在: {INPUT_CSV_PATH}")
        return

    # 读取CSV
    start_time = time.time()
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        print(f"✅ 读取CSV成功 | 行数: {len(df)} | 列名: {list(df.columns)}")

        if MAX_ROWS > 0 and len(df) > MAX_ROWS:
            df = df.head(MAX_ROWS)
            print(f"⚠️ 限制处理行数为 {MAX_ROWS}")

        all_data = [(idx + 2, row) for idx, row in df.iterrows()]

    except Exception as e:
        print(f"❌ 读取CSV失败: {e}")
        return

    # 并行处理
    all_results = []
    success_count = 0
    failed_count = 0
    total_rows = len(all_data)
    actual_max_workers = min(MAX_WORKERS, total_rows) if total_rows else 1

    print(f"\n📌 并行配置 | 线程数: {actual_max_workers} | 任务数: {total_rows}")

    if total_rows > 0:
        with concurrent.futures.ThreadPoolExecutor(max_workers=actual_max_workers) as executor:
            future_to_row = {executor.submit(process_single_row, data): data for data in all_data}

            for idx, future in enumerate(concurrent.futures.as_completed(future_to_row), start=1):
                try:
                    result = future.result()
                    all_results.append(result)

                    if result.get("status") == "success":
                        success_count += 1
                    else:
                        failed_count += 1

                    progress = (idx / total_rows) * 100
                    with lock:
                        print(f"\n📈 进度: {idx}/{total_rows} ({progress:.1f}%) | 成功: {success_count} | 失败: {failed_count}")

                except Exception as e:
                    failed_count += 1
                    with lock:
                        print(f"❌ 任务{idx}异常: {str(e)}")

    # 保存结果
    os.makedirs(RESULT_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_csv_path = os.path.join(RESULT_DIR, f"{BASE_FILENAME}_{timestamp}.csv")
    latest_csv_path = os.path.join(RESULT_DIR, f"{BASE_FILENAME}_latest.csv")

    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_FIELD_NAMES)
        csv_writer.writeheader()
        for result in all_results:
            write_data = {k: v for k, v in result.items() if k in OUTPUT_FIELD_NAMES}
            csv_writer.writerow(write_data)

    shutil.copy2(output_csv_path, latest_csv_path)
    print(f"\n✅ 结果文件: {output_csv_path}")
    print(f"✅ Latest文件: {latest_csv_path}")

    # 输出统计
    total_time = time.time() - start_time
    print("\n📋 SAR-VQA结果评估统计报告")
    print("=" * 80)
    print(f"⏱️ 总耗时: {int(total_time // 60)}分{total_time % 60:.2f}秒")
    print(f"📊 总行数: {len(all_results)}")
    print(f"✅ 成功: {success_count} | ❌ 失败: {failed_count}")
    print(f"📈 成功率: {(success_count / len(all_results) * 100):.2f}%" if all_results else "📈 成功率: 0%")

    if all_results:
        success_results = [r for r in all_results if r.get("status") == "success"]
        if success_results:
            avg_cosine = np.mean([r['cosine'] for r in success_results])
            avg_rouge = np.mean([r['ROUGEL'] for r in success_results])
            match_count = sum(1 for r in success_results if r['correct'] == '1')
            match_rate = match_count / len(success_results) if success_results else 0

            print(f"\n📝 语义匹配率: {match_rate:.4f} ({match_rate*100:.2f}%)")
            print(f"📝 平均余弦相似度: {avg_cosine:.4f}")
            print(f"📝 平均ROUGE-L: {avg_rouge:.4f}")

    print("\n🎉 评估完成！")
    print("=" * 80)


if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    main()
