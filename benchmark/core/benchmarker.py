"""
指标评估模块
包含 benchmark 评估流水线 + 所有相似度/语义匹配指标计算
"""

import csv
import os
import time
import shutil
import threading

import numpy as np
import pandas as pd
import requests
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from config import doubao_config

# ====================== 线程安全 ======================
LOCK = threading.Lock()

# ====================== 语义匹配 API 配置 ======================
SEMANTIC_API_KEY = doubao_config.API_KEY
SEMANTIC_API_ENDPOINT = doubao_config.API_ENDPOINT
SEMANTIC_MODEL_NAME = "doubao-1-5-lite-32k-250115"
SEMANTIC_API_TIMEOUT = 30
SEMANTIC_API_TEMPERATURE = 0.7

SEMANTIC_SYSTEM_PROMPT = "你是一个语义匹配判断助手，仅返回1或0，不解释原因"
SEMANTIC_PROMPT_TEMPLATE = """
Question: {question}
Ground Truth Answer: {answer}
Predicted Answer: {predicted}
Does the predicted answer match the ground truth? Answer 1 for match and 0 for not match. 
Use semantic meaning not exact match. Synonyms are also treated as a match, e.g., football and soccer, 
playground and ground track field, building and rooftop, pond and swimming pool. Do not explain the reason.
""".strip()


# ====================== 语义匹配 API 调用 ======================
def call_chat_api(question: str, answer: str, predicted: str) -> str:
    """
    调用豆包 API 判断预测答案是否与标准答案语义匹配
    返回 '1'（匹配）或 '0'（不匹配）
    """
    formatted_prompt = SEMANTIC_PROMPT_TEMPLATE.format(
        question=question, answer=answer, predicted=predicted
    )

    data = {
        "model": SEMANTIC_MODEL_NAME,
        "temperature": float(SEMANTIC_API_TEMPERATURE),
        "messages": [
            {"role": "system", "content": SEMANTIC_SYSTEM_PROMPT},
            {"role": "user", "content": formatted_prompt}
        ],
        "thinking": {"type": "disabled"}
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SEMANTIC_API_KEY}"
    }

    try:
        response = requests.post(
            SEMANTIC_API_ENDPOINT, json=data, headers=headers, timeout=SEMANTIC_API_TIMEOUT
        )

        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                message_content = result['choices'][0].get('message', {}).get('content', '').strip()
                if message_content not in ["0", "1"]:
                    message_content = "1" if "1" in message_content else "0" if "0" in message_content else "0"
                return message_content
            else:
                with LOCK:
                    print(f"API 返回无有效内容：{result}")
                return "0"
        else:
            with LOCK:
                print(f"API 调用失败，状态码：{response.status_code}")
                print(f"错误信息：{response.text}")
            return "0"

    except requests.exceptions.RequestException as e:
        with LOCK:
            print(f"API 请求异常：{e}")
        return "0"
    except Exception as e:
        with LOCK:
            print(f"处理 API 响应时发生错误：{e}")
        return "0"


# ====================== 相似度指标计算 ======================
def calculate_cosine_similarity(text1: str, text2: str) -> float:
    """计算两个文本的余弦相似度（TF-IDF 加权）"""
    if not text1 or not text2:
        return 0.0

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            stop_words='english',
            min_df=1
        )

        tfidf_matrix = vectorizer.fit_transform([text1.lower(), text2.lower()])

        if tfidf_matrix.nnz == 0:
            return 0.0

        return round(float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]), 4)

    except Exception as e:
        with LOCK:
            print(f"计算余弦相似度错误: {str(e)}")
        return 0.0


def calculate_rouge_l(text1: str, text2: str) -> float:
    """计算两个文本之间的 ROUGE-L 分数"""
    if not text1 or not text2:
        return 0.0

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(text1, text2)
        return scores['rougeL'].fmeasure
    except Exception as e:
        with LOCK:
            print(f"  [ROUGE-L 计算错误]: {str(e)}")
        return 0.0


def calculate_advanced_metrics(predicted: str, answer: str) -> Dict[str, float]:
    """计算 BLEU-1 ~ BLEU-4 和 METEOR 分数"""
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score

    results = {k: 0.0 for k in ['BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'METEOR']}

    if not predicted or not answer:
        return results

    pred_tokens = predicted.lower().split()
    ans_tokens = answer.lower().split()
    ans_tokens_list = [ans_tokens]

    smoothie = SmoothingFunction().method1
    weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]

    for i, w in enumerate(weights):
        results[f'BLEU{i + 1}'] = round(
            sentence_bleu(ans_tokens_list, pred_tokens, weights=w, smoothing_function=smoothie), 4
        )

    try:
        results['METEOR'] = round(meteor_score(ans_tokens_list, pred_tokens), 4)
    except Exception:
        results['METEOR'] = 0.0

    return results


def calculate_all_metrics(predicted: str, answer: str) -> dict:
    """计算所有评估指标（cosine + ROUGE-L + BLEU + METEOR）"""
    results = {
        'cosine': calculate_cosine_similarity(predicted, answer),
        'ROUGEL': calculate_rouge_l(predicted, answer),
    }
    results.update(calculate_advanced_metrics(predicted, answer))
    return results


# ====================== 单行处理 ======================
OUTPUT_FIELD_NAMES = [
    'id', 'image', 'question', 'answer', 'predicted',
    'correct', 'cosine', 'ROUGEL', 'BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'METEOR', 'IG', 'ID'
]


def process_single_row(row_data: tuple) -> dict:
    """
    处理单行数据：计算相似度 + 调用 API 判断匹配度

    row_data 格式：(row_num, row)
    """
    try:
        row_num, row = row_data

        id_str = str(row['id']) if pd.notna(row['id']) else ""
        image_path = str(row['image']) if pd.notna(row['image']) else ""
        question = str(row['question']) if pd.notna(row['question']) else ""
        answer = str(row['answer']) if pd.notna(row['answer']) else ""
        predicted = str(row['predicted']) if pd.notna(row['predicted']) else ""
        ig_val = float(row['IG']) if pd.notna(row['IG']) else 0.0
        id_val = float(row['ID']) if pd.notna(row['ID']) else 0.0

        with LOCK:
            print(f" 正在处理行 {row_num} | ID：{id_str} | 问题摘要：{question[:50]}...")

        cosine_sim = calculate_cosine_similarity(predicted, answer)
        rouge_l = calculate_rouge_l(predicted, answer)
        correct = call_chat_api(question, answer, predicted)
        adv_metrics = calculate_advanced_metrics(predicted, answer)

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
            "status": "success",
            "IG": ig_val,
            "ID": id_val,
        }

        with LOCK:
            print(f" 行 {row_num} 处理完成 | 匹配结果：{correct} | 余弦相似度：{cosine_sim:.4f} | ROUGE-L：{rouge_l:.4f}")
        return result

    except Exception as e:
        error_msg = f" 行 {row_num} 处理失败：{str(e)}"
        with LOCK:
            print(error_msg)
        return {
            "id": "", "image": "", "question": "", "answer": "", "predicted": "",
            "correct": "0", "cosine": 0.0, "ROUGEL": 0.0,
            "BLEU1": 0.0, "BLEU2": 0.0, "BLEU3": 0.0, "BLEU4": 0.0, "METEOR": 0.0,
            "status": "failed", "IG": 0.0, "ID": 0.0, "error": error_msg
        }


# ====================== Benchmark 评估流水线 ======================
class Benchmarker:
    """Benchmark 评估器：并行评估 VQA 预测结果"""

    def __init__(
        self,
        input_csv_path: str,
        result_dir: str = "./benchmark/result/benchmark_result/",
        base_filename: str = "benchmark",
        max_rows: int = 50000,
        max_workers: int = 30
    ):
        self.input_csv_path = input_csv_path
        self.result_dir = result_dir
        self.base_filename = base_filename
        self.max_rows = max_rows
        self.max_workers = max_workers
        self.results = []
        self.success_count = 0
        self.failed_count = 0
        self.total_time = 0.0

    def run(self) -> str:
        """
        执行 benchmark 评估流水线

        返回:
            str: 输出 CSV 路径
        """
        start_time = time.time()

        with LOCK:
            print("=" * 80)
            print("开始执行 VQA 结果评估流程（SAR-VQA 数据集）")
            print("=" * 80)

        # 1. 读取输入
        with LOCK:
            print("\n[步骤 1/4] 读取输入 CSV 文件...")

        if not os.path.exists(self.input_csv_path):
            with LOCK:
                print(f"错误：输入文件 {self.input_csv_path} 不存在")
            return None

        try:
            df = pd.read_csv(self.input_csv_path)
            with LOCK:
                print(f"成功读取 CSV 文件 | 总行数：{len(df)} | 列名：{list(df.columns)}")

            if self.max_rows > 0 and len(df) > self.max_rows:
                df = df.head(self.max_rows)
                with LOCK:
                    print(f"已限制处理行数为前 {self.max_rows} 行")

            all_data = [(idx + 2, row) for idx, row in df.iterrows()]

        except Exception as e:
            with LOCK:
                print(f"读取 CSV 文件失败：{e}")
            return None

        total_rows = len(all_data)
        with LOCK:
            print(f"完成 CSV 文件读取 | 有效数据行数：{total_rows}")

        # 2. 并行处理
        with LOCK:
            print("\n[步骤 2/4] 启动并行处理任务...")

        actual_workers = min(self.max_workers, total_rows) if total_rows else 1
        with LOCK:
            print(f"并行配置 | 最大线程数：{actual_workers} | 待处理任务数：{total_rows}")

        if total_rows == 0:
            with LOCK:
                print("无有效数据可处理")
        else:
            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                future_to_row = {executor.submit(process_single_row, data): data for data in all_data}

                with LOCK:
                    print(f"\n开始处理任务（共 {len(future_to_row)} 个），请等待...")

                for idx, future in enumerate(concurrent.futures.as_completed(future_to_row), start=1):
                    try:
                        result = future.result()
                        self.results.append(result)

                        if result.get("status") == "success":
                            self.success_count += 1
                        else:
                            self.failed_count += 1

                        progress = (idx / total_rows) * 100
                        with LOCK:
                            print(f"\n处理进度：{idx}/{total_rows} ({progress:.1f}%) | 成功：{self.success_count} | 失败：{self.failed_count}")

                    except Exception as e:
                        self.failed_count += 1
                        with LOCK:
                            print(f"任务 {idx} 执行异常：{str(e)}")
                        self.results.append({"status": "error", "error": str(e)})

        # 3. 写入结果
        with LOCK:
            print("\n[步骤 3/4] 写入评估结果到 CSV 文件...")

        os.makedirs(self.result_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        output_csv_path = os.path.join(self.result_dir, f"{self.base_filename}_{timestamp}.csv")
        latest_csv_path = os.path.join(self.result_dir, f"{self.base_filename}_latest.csv")

        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_FIELD_NAMES)
            csv_writer.writeheader()
            for result in self.results:
                write_data = {k: v for k, v in result.items() if k in OUTPUT_FIELD_NAMES}
                csv_writer.writerow(write_data)

        with LOCK:
            print(f"带时间戳的结果文件写入完成 | 路径：{output_csv_path}")

        try:
            shutil.copy2(output_csv_path, latest_csv_path)
            with LOCK:
                print(f"Latest 版本文件生成完成 | 路径：{latest_csv_path}")
        except Exception as e:
            with LOCK:
                print(f"Latest 版本文件复制失败：{str(e)}")

        # 4. 统计报告
        self._print_report(start_time)

        self.total_time = time.time() - start_time
        return output_csv_path

    def _print_report(self, start_time: float):
        """打印评估统计报告"""
        total_time = time.time() - start_time
        minutes = int(total_time // 60)
        seconds = total_time % 60

        with LOCK:
            print("\n[步骤 4/4] 生成评估统计报告...")
            print("=" * 80)
            print("SAR-VQA 结果评估统计报告")
            print("=" * 80)
            print(f"总执行耗时：{minutes}分{seconds:.2f}秒")
            print(f"总处理数据行数：{len(self.results)}")
            print(f"成功处理行数：{self.success_count}")
            print(f"失败处理行数：{self.failed_count}")
            if self.results:
                print(f"处理成功率：{(self.success_count / len(self.results) * 100):.2f}%")

        if self.results:
            success_results = [r for r in self.results if r.get("status") == "success"]
            if success_results:
                avg_cosine = np.mean([r['cosine'] for r in success_results])
                avg_rouge = np.mean([r['ROUGEL'] for r in success_results])
                avg_ig = np.mean([r['IG'] for r in success_results])
                avg_id = np.mean([r['ID'] for r in success_results])
                match_count = sum(1 for r in success_results if r['correct'] == '1')
                match_rate = match_count / len(success_results) if success_results else 0

                with LOCK:
                    print(f"\n语义匹配统计（仅成功行）：")
                    print(f"   - 匹配数：{match_count} | 总成功数：{len(success_results)}")
                    print(f"   - 语义匹配率：{match_rate:.4f} ({match_rate * 100:.2f}%)")
                    print(f"\n相似度统计（仅成功行）：")
                    print(f"   - 平均余弦相似度：{avg_cosine:.4f}")
                    print(f"   - 平均 ROUGE-L 分数：{avg_rouge:.4f}")
                    print(f"   - 平均信息增益度 IG：{avg_ig:.4f}")
                    print(f"   - 平均信息密度 ID：{avg_id:.4f}")
            else:
                with LOCK:
                    print("\n无成功处理的数据，无法计算相似度和匹配率")

        with LOCK:
            print("\n整个评估流程执行完成！")
            print("=" * 80)

    def check_input_csv_structure(self) -> bool:
        """检查输入 CSV 文件的结构和列名"""
        try:
            with open(self.input_csv_path, mode='r', newline='', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                headers = next(csv_reader)
                print("输入 CSV 文件结构检查:")
                print(f"列名：{headers}")
                print(f"预期最少列数：5 (id, image, question, answer, predicted)")
                print(f"实际列数：{len(headers)}")

                print("\n前 5 行数据预览:")
                for i, row in enumerate(csv_reader):
                    if i >= 5:
                        break
                    print(f"行 {i + 2}: {row[:7]}")
            return True
        except Exception as e:
            print(f"检查 CSV 文件结构失败：{e}")
            return False
