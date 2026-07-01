import csv
import json
import os
import time
import shutil
import numpy as np
import threading
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import requests
from typing import Optional, List, Dict
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import math
import pandas as pd

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
# 如果环境中没有 nltk 数据，建议在 main 前运行：nltk.download('punkt')

# -------------------------- 全局可配置常量 --------------------------
# ====================== 线程安全配置 ======================
LOCK = threading.Lock()

# ====================== 大模型配置 ======================
VOLC_API_KEY = "e9f2dacd-d0a2-4c9a-ba2a-805ee0b40dcd"
VOLC_API_ENDPOINT = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
VOLC_MODEL_NAME = "doubao-1-5-lite-32k-250115"
# VOLC_MODEL_NAME = "doubao-seed-1-6-flash-250828"  # 另一个可选模型
API_TIMEOUT = 30  # API请求超时时间（秒）
API_TEMPERATURE = 0.7  # API温度参数
API_SESSION_ID = "test_session_1"  # API会话ID
API_SYSTEM_PROMPT = "你是一个语义匹配判断助手，仅返回1或0，不解释原因"
API_PROMPT_TEMPLATE = """
Question: {question}
Ground Truth Answer: {answer}
Predicted Answer: {predicted}
Does the predicted answer match the ground truth? Answer 1 for match and 0 for not match. 
Use semantic meaning not exact match. Synonyms are also treated as a match, e.g., football and soccer, 
playground and ground track field, building and rooftop, pond and swimming pool. Do not explain the reason.
""".strip()

# ====================== 文件路径配置 ======================
# FILE_TAG = "doubao-seed-2-0-mini-260428"  # 文件标识（可选：original、optimized等）
# FILE_TAG = "internvl2-8b"  # 文件标识（可选：original、optimized等）
# FILE_TAG = "internvl3_5-8b"  # 文件标识（可选：original、optimized等）

# FILE_TAG = "agent-text-doubao-seed-2-0-mini"  # 文件标识（可选：original、optimized等）
FILE_TAG = "agent-text-doubao-seed-2-0-mini_rscsv"  # 文件标识（可选：original、optimized等）
# FILE_TAG = "agent-text-internVL2-8b"  # 文件标识（可选：original、optimized等）
# FILE_TAG = "agent-text-internVL3_5-8b"  # 文件标识（可选：original、optimized等）
# DATASET_TAG = "test"
DATASET_TAG = "val"
INPUT_CSV_PATH = f'./benchmark/result/{FILE_TAG}/{FILE_TAG}_{DATASET_TAG}_predicted_question_latest.csv'
# INPUT_CSV_PATH = f'./benchmark/result/{FILE_TAG}/doubao-seed-2-0-mini-260428_val_predicted_question_20260624_223032.csv'
RESULT_DIR = "./benchmark/result/benchmark_result/"
BASE_FILENAME = f"{FILE_TAG}_{DATASET_TAG}_benchmark"
# ---------------------------------------------------------------------------------
# ====================== 数据处理配置 ======================
MAX_ROWS = 50000  # 最大处理行数
MAX_WORKERS = 30  # 最大并发线程数（根据API限流调整）
STOP_WORDS = [
    'the', 'a', 'an', 'and', 'or', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 
    'to', 'for', 'of', 'with', 'by', 'this', 'that', 'these', 'those', 'it', 'its'
]  # 停用词列表
NGRAM_RANGE = (1, 3)  # ngram范围
MIN_DF = 1  # 最小文档频率
TOKEN_PATTERN = r'\b[a-zA-Z0-9\-]+\b'  # 分词正则表达式
OUTPUT_FIELD_NAMES = [
    'id', 'image', 'question','answer', 'predicted', 
    'correct', 'cosine', 'ROUGEL', 'BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'METEOR', 'IG', 'ID'
]

# ====================== 指标计算函数 ======================
def calculate_advanced_metrics(predicted: str, answer: str) -> Dict[str, float]:
    results = {k: 0.0 for k in ['BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'METEOR']}
    if not predicted or not answer:
        return results

    pred_tokens = predicted.lower().split()
    ans_tokens = answer.lower().split()
    ans_tokens_list = [ans_tokens]

    # 1. BLEU (带平滑)
    smoothie = SmoothingFunction().method1
    weights = [(1,0,0,0), (0.5,0.5,0,0), (0.33,0.33,0.33,0), (0.25,0.25,0.25,0.25)]
    for i, w in enumerate(weights):
        results[f'BLEU{i+1}'] = round(sentence_bleu(ans_tokens_list, pred_tokens, weights=w, smoothing_function=smoothie), 4)

    # 2. METEOR
    try:
        results['METEOR'] = round(meteor_score(ans_tokens_list, pred_tokens), 4)
    except:
        results['METEOR'] = 0.0

    return results

# ===================== 功能函数 =====================
def call_chat_api(question: str, answer: str, predicted: str):
    """
    调用豆包API接口判断预测答案是否与标准答案语义匹配
    返回 '1'（匹配）或 '0'（不匹配）
    """
    # 构建提示词
    formatted_prompt = API_PROMPT_TEMPLATE.format(
        question=question,
        answer=answer,
        predicted=predicted
    )
    
    # 构建请求数据
    data = {
        "model": VOLC_MODEL_NAME,
        "temperature": float(API_TEMPERATURE),
        "messages": [
            {
                "role": "system",
                "content": API_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": formatted_prompt
            }
        ],
        "thinking": {
            "type": "disabled"
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {VOLC_API_KEY}"
    }
    
    try:
        # 发送POST请求
        response = requests.post(
            VOLC_API_ENDPOINT, 
            json=data, 
            headers=headers, 
            timeout=API_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            # 提取回复内容
            if 'choices' in result and len(result['choices']) > 0:
                message_content = result['choices'][0].get('message', {}).get('content', '').strip()
                # 确保返回值是1或0
                if message_content not in ["0", "1"]:
                    message_content = "1" if "1" in message_content else "0" if "0" in message_content else "0"
                return message_content
            else:
                with LOCK:
                    print(f"API返回无有效内容：{result}")
                return "0"
        else:
            with LOCK:
                print(f"API调用失败，状态码：{response.status_code}")
                print(f"错误信息：{response.text}")
            return "0"
            
    except requests.exceptions.RequestException as e:
        with LOCK:
            print(f"API请求异常：{e}")
        return "0"
    except Exception as e:
        with LOCK:
            print(f"处理API响应时发生错误：{e}")
        return "0"

def calculate_cosine_similarity(text1: str, text2: str) -> float:
    """计算两个文本的余弦相似度"""
    if not text1 or not text2:
        with LOCK:
            print(f"⚠️  空文本：text1={text1[:20] if text1 else '空'}, text2={text2[:20] if text2 else '空'}")
        return 0.0
    
    # 打印原始文本前50字符（确认文本读取正常）
    with LOCK:
        print(f"📝 原始文本1：{text1[:50]}...")
        print(f"📝 原始文本2：{text2[:50]}...")
    
    try:
        # 初始化TF-IDF向量化器
        vectorizer = TfidfVectorizer(
            ngram_range=NGRAM_RANGE,
            stop_words=STOP_WORDS,
            min_df=MIN_DF,
            token_pattern=TOKEN_PATTERN
        )
        
        # 转换文本并打印词汇表（确认特征提取正常）
        tfidf_matrix = vectorizer.fit_transform([text1.lower(), text2.lower()])
        vocab = vectorizer.get_feature_names_out()
        
        with LOCK:
            print(f"🔤 提取的词汇表：{vocab[:10]}..." if len(vocab) > 0 else "❌ 未提取到任何词汇")
        
        if tfidf_matrix.nnz == 0:
            with LOCK:
                print("❌ TF-IDF矩阵全为0，无有效特征")
            return 0.0
        
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        with LOCK:
            print(f"✅ 相似度结果：{cosine_sim:.4f}")
        
        return round(cosine_sim, 4)
    except Exception as e:
        with LOCK:
            print(f"❌ 计算错误：{str(e)}")
        return 0.0

def calculate_rouge_l(text1: str, text2: str) -> float:
    """
    计算两个文本之间的ROUGE-L分数（text1: predicted, text2: answer）
    """
    if not text1 or not text2:
        return 0.0
    
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(text1, text2)
        return scores['rougeL'].fmeasure
    except Exception as e:
        with LOCK:
            print(f"  [ROUGE-L计算错误]: {str(e)}")
        return 0.0

def process_single_row(row_data: tuple):
    """
    处理单行数据：计算相似度 + 调用API判断匹配度
    修改点：不再接收 header_map，直接通过 row[column] 读取
    row_data格式：(row_num, row)
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

        # 打印当前处理行的基础信息（加锁保证线程安全）
        with LOCK:
            print(f" 正在处理行 {row_num} | ID：{id_str} | 问题摘要：{question[:50]}...")

        # 计算相似度指标（predicted相对answer）
        cosine_sim = calculate_cosine_similarity(predicted, answer)
        rouge_l = calculate_rouge_l(predicted, answer)

        # 调用API判断匹配度
        correct = call_chat_api(question, answer, predicted)

        # 新增高级指标
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
        # 返回空值兜底
        return {
            "id": "",
            "image": "",
            "question": "",
            "answer": "",
            "predicted": "",
            "correct": "0",
            "cosine": 0.0,
            "ROUGEL": 0.0,
            "BLEU1": 0.0,
            "BLEU2": 0.0,
            "BLEU3": 0.0,
            "BLEU4": 0.0,
            "METEOR": 0.0,
            "status": "failed",
            "IG": 0.0,
            "ID": 0.0,
            "error": error_msg
        }

def process_cosine_rouge_with_limit():
    """
    处理VQA结果评估（并行版本）
    使用常量配置区的参数
    """
    # 记录整体开始时间
    start_time = time.time()
    with LOCK:
        print("="*80)
        print("开始执行VQA结果评估流程（SAR-VQA数据集）")
        print("="*80)

    # 1. 读取CSV文件
    with LOCK:
        print("\n[步骤1/4] 读取输入CSV文件...")

    if not os.path.exists(INPUT_CSV_PATH):
        with LOCK:
            print(f"❌ 错误：输入文件 {INPUT_CSV_PATH} 不存在，程序退出")
        return

    try:
        # 读取CSV
        df = pd.read_csv(INPUT_CSV_PATH)
        with LOCK:
            print(f"✅ 成功读取CSV文件 | 总行数：{len(df)} | 列名：{list(df.columns)}")

        # 限制行数
        if MAX_ROWS > 0 and len(df) > MAX_ROWS:
            df = df.head(MAX_ROWS)
            with LOCK:
                print(f"⚠️ 已限制处理行数为前 {MAX_ROWS} 行")

        # 将 DataFrame 转换为 (行号, 行数据) 的列表
        # iterrows() 返回的是 (index, Series) 元组
        all_data = [(idx+2, row) for idx, row in df.iterrows()] # idx+2 是为了匹配原代码的行号逻辑（跳过表头和从2开始计数）

    except Exception as e:
        with LOCK:
            print(f"❌ 读取CSV文件失败：{e}")
        return
    
    # 打印数据读取统计
    total_rows = len(all_data)
    with LOCK:
        print(f"✅ 完成CSV文件读取 | 有效数据行数：{total_rows} | 最大限制行数：{MAX_ROWS}")

    # 2. 并行处理所有数据
    with LOCK:
        print("\n[步骤2/4] 启动并行处理任务...")
    all_results = []
    success_count = 0
    failed_count = 0
    
    # 设置最大线程数
    actual_max_workers = min(MAX_WORKERS, total_rows) if total_rows else 1
    with LOCK:
        print(f"📌 并行配置 | 最大线程数：{actual_max_workers} | 待处理任务数：{total_rows}")
    
    if total_rows == 0:
        with LOCK:
            print("⚠️  无有效数据可处理，跳过并行步骤")
    else:
        with ThreadPoolExecutor(max_workers=actual_max_workers) as executor:
            # 提交所有任务，同时传入 header_map
            future_to_row = {executor.submit(process_single_row, data): data for data in all_data}
            
            # 收集结果并统计进度
            with LOCK:
                print(f"\n📊 开始处理任务（共{len(future_to_row)}个），请等待...")
            
            for idx, future in enumerate(concurrent.futures.as_completed(future_to_row), start=1):
                try:
                    result = future.result()
                    all_results.append(result)
                    
                    # 统计成功/失败数
                    if result.get("status") == "success":
                        success_count += 1
                    else:
                        failed_count += 1
                    
                    # 打印进度（加锁保证线程安全）
                    progress = (idx / total_rows) * 100
                    with LOCK:
                        print(f"\n📈 处理进度：{idx}/{total_rows} ({progress:.1f}%) | 成功：{success_count} | 失败：{failed_count}")
                
                except Exception as e:
                    failed_count += 1
                    error_msg = f"❌ 任务{idx}执行异常：{str(e)}"
                    with LOCK:
                        print(error_msg)
                    all_results.append({
                        "status": "error",
                        "error": error_msg,
                        "success": False
                    })
    
    # 3. 写入结果到CSV文件
    with LOCK:
        print("\n[步骤3/4] 写入评估结果到CSV文件...")
    
    # 确保结果目录存在
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # 生成时间戳和文件路径
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_csv_path = os.path.join(RESULT_DIR, f"{BASE_FILENAME}_{timestamp}.csv")
    latest_csv_path = os.path.join(RESULT_DIR, f"{BASE_FILENAME}_latest.csv")
    
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_FIELD_NAMES)
        csv_writer.writeheader()
        for result in all_results:
            # 过滤掉不需要的字段，只写入定义的fieldnames
            write_data = {k: v for k, v in result.items() if k in OUTPUT_FIELD_NAMES}
            csv_writer.writerow(write_data)
    
    with LOCK:
        print(f"✅ 带时间戳的结果文件写入完成 | 路径：{output_csv_path}")
    
    # 复制生成latest文件
    try:
        shutil.copy2(output_csv_path, latest_csv_path)
        with LOCK:
            print(f"✅ Latest版本文件生成完成 | 路径：{latest_csv_path}")
    except Exception as e:
        with LOCK:
            print(f"⚠️  Latest版本文件复制失败：{str(e)}")

    # 4. 计算并打印统计信息
    with LOCK:
        print("\n[步骤4/4] 生成评估统计报告...")
        print("="*80)
        print("📋 SAR-VQA结果评估统计报告")
        print("="*80)
    
    # 计算总耗时
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = total_time % 60
    
    # 基础统计
    with LOCK:
        print(f"⏱️  总执行耗时：{minutes}分{seconds:.2f}秒")
        print(f"📊 总处理数据行数：{len(all_results)}")
        print(f"✅ 成功处理行数：{success_count}")
        print(f"❌ 失败处理行数：{failed_count}")
        print(f"📈 处理成功率：{(success_count / len(all_results) * 100):.2f}%" if all_results else "📈 处理成功率：0%")
    
    # 相似度统计（仅针对成功处理的数据）
    if all_results:
        success_results = [r for r in all_results if r.get("status") == "success"]
        if success_results:
            avg_cosine = np.mean([r['cosine'] for r in success_results])
            avg_rouge = np.mean([r['ROUGEL'] for r in success_results])
            avg_ig = np.mean([r['IG'] for r in success_results])
            avg_id = np.mean([r['ID'] for r in success_results])
            # 计算匹配率
            match_count = sum(1 for r in success_results if r['correct'] == '1')
            match_rate = match_count / len(success_results) if success_results else 0
            
            with LOCK:
                print(f"\n📝 语义匹配统计（仅成功行）：")
                print(f"   - 匹配数：{match_count} | 总成功数：{len(success_results)}")
                print(f"   - 语义匹配率：{match_rate:.4f} ({match_rate*100:.2f}%)")
                print(f"\n📝 相似度统计（仅成功行）：")
                print(f"   - 平均余弦相似度：{avg_cosine:.4f}")
                print(f"   - 平均ROUGE-L分数：{avg_rouge:.4f}")
                print(f"   - 平均信息增益度 IG：{avg_ig:.4f}")
                print(f"   - 平均信息密度 ID：{avg_id:.4f}")
        else:
            with LOCK:
                print("\n⚠️  无成功处理的数据，无法计算相似度和匹配率")
    else:
        with LOCK:
            print("\n⚠️  无处理结果，无法生成统计信息")
    
    with LOCK:
        print("\n🎉 整个评估流程执行完成！")
        print("="*80)

def check_input_csv_structure():
    """
    检查输入CSV文件的结构和列名
    """
    try:
        with open(INPUT_CSV_PATH, mode='r', newline='', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            headers = next(csv_reader)
            print("📊 输入CSV文件结构检查:")
            print(f"列名：{headers}")
            print(f"预期最少列数：5 (id, image, question, answer, predicted)")
            print(f"实际列数：{len(headers)}")
            
            # 读取前5行数据预览
            print("\n前5行数据预览:")
            for i, row in enumerate(csv_reader):
                if i >= 5:
                    break
                print(f"行{i+2}: {row[:7]}")  # 只显示前7列
        return True
    except Exception as e:
        print(f"❌ 检查CSV文件结构失败：{e}")
        return False

def main():
    """
    主函数
    """
    print("🚀 开始并行处理SAR-VQA结果评估...")
    
    # 前置检查
    if not os.path.exists(os.path.dirname(INPUT_CSV_PATH)):
        print(f"❌ 错误：输入文件目录不存在：{os.path.dirname(INPUT_CSV_PATH)}")
        return
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"❌ 错误：输入CSV文件不存在：{INPUT_CSV_PATH}")
        return
    if not check_input_csv_structure():
        print("❌ 输入CSV文件结构检查失败，请检查文件格式")
        return
    
    # 并行处理
    process_cosine_rouge_with_limit()
    print("\n🏁 全部评估流程完成！")

if __name__ == "__main__":
    # 必须下载 wordnet 用于 METEOR 计算中的同义词匹配
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    main()