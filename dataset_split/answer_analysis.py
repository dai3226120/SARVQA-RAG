import csv
import os
import time
import threading
import pandas as pd
import requests
import json
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from collections import Counter
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.path_tool import get_abs_path

# -------------------------- 全局配置 --------------------------
from dotenv import load_dotenv
load_dotenv(get_abs_path(".env"))

# 1. API配置（从 .env 获取）
VOLC_API_KEY = os.environ.get('DOUBAO_SEED_API_KEY', '')
VOLC_API_ENDPOINT = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
VOLC_MODEL_NAME = "doubao-1-5-lite-32k-250115"
API_TIMEOUT = 30
API_TEMPERATURE = float(os.environ.get('DOUBAO_SEED_TEMPERATURE', 0.05))  # 从 .env 获取温度
API_MAX_RETRIES = 3
API_RETRY_DELAY = 1
API_SYSTEM_PROMPT = "你是严格的术语统计工具，只统计列表中明确列出的术语，不做任何扩展，仅返回JSON"

# 2. 核心优化：基于你的真实数据集定制的术语列表
# 完全删除了未出现的学术术语，只保留实际存在和可能出现的术语
API_PROMPT_TEMPLATE = """
请严格统计以下答案文本中出现的SAR专业术语数量，**只统计下面列表中明确列出的术语**，不要自行添加任何其他术语。

### 统计分类及精确术语列表（大小写不敏感）
1. **SAR成像参数**（系统本身特性）：
   resolution（分辨率）、polarization（极化）、HH、HV、VH、VV、band（波段）、incidence angle（入射角）

2. **SAR散射特性**（地物与雷达波相互作用）：
   radar backscatter（雷达后向散射）、backscatter（后向散射）、backscatter patterns（后向散射模式）、
   speckled texture（斑点纹理）、coherent speckle（相干斑）、bright speckles（亮斑）、
   radar returns（雷达回波）、returns（回波）、radar signal（雷达信号）、
   blurred artifacts（模糊伪影）、smeared artifacts（拖尾伪影）

3. **SAR基础概念**（领域通用专业术语）：
   SAR image（SAR图像）、SAR imaging（SAR成像）、radar data（雷达数据）、
   radar（雷达）、ground truth（地面真值）、radar response（雷达响应）

### 统计规则
- ✅ 同一术语重复出现**只计数1次**
- ✅ 大小写不敏感（如"Radar Backscatter"和"radar backscatter"算同一个）
- ✅ 进行语义推断，语义匹配也算

Answer文本：{answer}

必须严格按照以下JSON格式返回，只返回JSON字符串，不要添加任何其他内容：
{{
    "imaging_params_count": 0,
    "scattering_features_count": 0,
    "basic_concepts_count": 0
}}
""".strip()

# 3. 文件路径配置
INPUT_CSV_PATH = get_abs_path("dataset_split/SAR-VQA_ALL-180375_filtered_dataset-SAR-ship_answer-empty.csv")
RESULT_DIR = get_abs_path("dataset_split/analysis_result")
RESULT_FILENAME = "sar_answer_analysis_final.csv"
STATISTICS_FILENAME = "sar_answer_statistics_final.txt"

# 4. 处理配置
MAX_ROWS = 200000
MAX_WORKERS = 100
LOCK = threading.Lock()

# 5. 输出字段（新增基础概念统计）
OUTPUT_FIELDS = [
    "id", "image", "question", "answer",
    "imaging_params_count", "scattering_features_count", "basic_concepts_count",
    "total_sar_terms_count", "status", "error"
]


# -------------------------- 核心函数 --------------------------
def call_sar_analysis_api(answer: str) -> Dict[str, int]:
    """
    调用API统计术语，增加严格的结果校验
    """
    if not answer or str(answer).strip() == "" or answer.strip() == "Answer":
        return {"imaging_params_count": 0, "scattering_features_count": 0, "basic_concepts_count": 0}

    formatted_prompt = API_PROMPT_TEMPLATE.format(answer=answer.strip())
    data = {
        "model": VOLC_MODEL_NAME,
        "temperature": float(API_TEMPERATURE),
        "messages": [
            {"role": "system", "content": API_SYSTEM_PROMPT},
            {"role": "user", "content": formatted_prompt}
        ],
        "thinking": {"type": "disabled"}
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {VOLC_API_KEY}"
    }

    for retry in range(API_MAX_RETRIES):
        try:
            response = requests.post(
                VOLC_API_ENDPOINT,
                json=data,
                headers=headers,
                timeout=API_TIMEOUT
            )

            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"].strip()
                    
                    try:
                        analysis_result = json.loads(content)
                        # 严格校验返回值范围（防止模型返回异常数字）
                        imaging = max(0, min(int(analysis_result.get("imaging_params_count", 0)), 8))
                        scattering = max(0, min(int(analysis_result.get("scattering_features_count", 0)), 10))
                        basic = max(0, min(int(analysis_result.get("basic_concepts_count", 0)), 5))
                        return {
                            "imaging_params_count": imaging,
                            "scattering_features_count": scattering,
                            "basic_concepts_count": basic
                        }
                    except json.JSONDecodeError:
                        import re
                        json_match = re.search(r'\{.*?\}', content, re.DOTALL)
                        if json_match:
                            try:
                                analysis_result = json.loads(json_match.group())
                                imaging = max(0, min(int(analysis_result.get("imaging_params_count", 0)), 8))
                                scattering = max(0, min(int(analysis_result.get("scattering_features_count", 0)), 10))
                                basic = max(0, min(int(analysis_result.get("basic_concepts_count", 0)), 5))
                                return {
                                    "imaging_params_count": imaging,
                                    "scattering_features_count": scattering,
                                    "basic_concepts_count": basic
                                }
                            except:
                                pass
                        
                        with LOCK:
                            print(f"⚠️  API返回格式错误：{content[:80]}")
                        continue
            else:
                with LOCK:
                    print(f"⚠️  API错误 {response.status_code}，重试 {retry+1}/{API_MAX_RETRIES}")
                time.sleep(API_RETRY_DELAY)
                continue

        except Exception as e:
            with LOCK:
                print(f"⚠️  请求异常：{str(e)}，重试 {retry+1}/{API_MAX_RETRIES}")
            time.sleep(API_RETRY_DELAY)
            continue

    with LOCK:
        print(f"❌  API调用失败，答案：{answer[:40]}...")
    return {"imaging_params_count": 0, "scattering_features_count": 0, "basic_concepts_count": 0}


def process_single_row(row_data: tuple) -> Dict:
    """
    处理单行数据，兼容你的CSV格式（包含image和question列）
    """
    try:
        row_num, row = row_data
        # 完全适配你的CSV列名
        id_str = str(row_num)
        image = str(row.get("image", ""))
        question = str(row.get("question", ""))
        answer = str(row.get("answer", ""))

        with LOCK:
            print(f"处理行 {row_num} | 答案：{answer[:50]}...")

        analysis_result = call_sar_analysis_api(answer)
        imaging = analysis_result["imaging_params_count"]
        scattering = analysis_result["scattering_features_count"]
        basic = analysis_result["basic_concepts_count"]
        total = imaging + scattering + basic

        result = {
            "id": id_str,
            "image": image,
            "question": question,
            "answer": answer[:1000],
            "imaging_params_count": imaging,
            "scattering_features_count": scattering,
            "basic_concepts_count": basic,
            "total_sar_terms_count": total,
            "status": "success",
            "error": ""
        }

        with LOCK:
            print(f"✅ 行 {row_num} | 成像参数：{imaging} | 散射特性：{scattering} | 基础概念：{basic} | 总计：{total}")
        return result

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        with LOCK:
            print(f"❌ 行 {row_num} 失败：{str(e)}")
        return {
            "id": str(row_num),
            "image": str(row_data[1].get("image", "")),
            "question": str(row_data[1].get("question", "")),
            "answer": str(row_data[1].get("answer", "")),
            "imaging_params_count": 0,
            "scattering_features_count": 0,
            "basic_concepts_count": 0,
            "total_sar_terms_count": 0,
            "status": "failed",
            "error": error_msg
        }


def batch_process_answers() -> List[Dict]:
    """
    批量处理函数
    """
    start_time = time.time()
    with LOCK:
        print("="*80)
        print("SAR数据集专业术语统计分析（适配真实数据版）")
        print(f"并发数：{MAX_WORKERS} | 最大重试：{API_MAX_RETRIES}")
        print("="*80)

    if not os.path.exists(INPUT_CSV_PATH):
        with LOCK:
            print(f"❌ 输入文件不存在：{INPUT_CSV_PATH}")
        return []

    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        with LOCK:
            print(f"✅ 读取CSV成功 | 总行数：{len(df)} | 列名：{list(df.columns)}")

        # 检查必要列
        required_cols = ["answer"]
        for col in required_cols:
            if col not in df.columns:
                with LOCK:
                    print(f"❌ 缺少必要列：{col}")
                return []

        if MAX_ROWS > 0 and len(df) > MAX_ROWS:
            df = df.head(MAX_ROWS)
            with LOCK:
                print(f"⚠️  限制处理前 {MAX_ROWS} 行")

        all_data = [(idx + 2, row) for idx, row in df.iterrows()]

    except Exception as e:
        with LOCK:
            print(f"❌ 读取CSV失败：{e}")
        return []

    all_results = []
    success_count = 0
    failed_count = 0
    total_rows = len(all_data)

    if total_rows == 0:
        with LOCK:
            print("⚠️  无有效数据")
        return []

    actual_max_workers = min(MAX_WORKERS, total_rows)
    with LOCK:
        print(f"🚀 启动处理 | 总任务：{total_rows}")

    with ThreadPoolExecutor(max_workers=actual_max_workers) as executor:
        future_to_row = {executor.submit(process_single_row, data): data for data in all_data}

        for idx, future in enumerate(future_to_row, start=1):
            try:
                result = future.result()
                all_results.append(result)
                if result["status"] == "success":
                    success_count += 1
                else:
                    failed_count += 1

                progress = (idx / total_rows) * 100
                with LOCK:
                    print(f"\n📊 进度：{idx}/{total_rows} ({progress:.1f}%) | 成功：{success_count} | 失败：{failed_count}")

            except Exception as e:
                failed_count += 1
                error_msg = f"任务{idx}异常：{str(e)}"
                with LOCK:
                    print(error_msg)
                all_results.append({
                    "id": str(idx),
                    "image": "",
                    "question": "",
                    "answer": "",
                    "imaging_params_count": 0,
                    "scattering_features_count": 0,
                    "basic_concepts_count": 0,
                    "total_sar_terms_count": 0,
                    "status": "error",
                    "error": error_msg
                })

    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = total_time % 60
    with LOCK:
        print(f"\n⏱️  总耗时：{minutes}分{seconds:.2f}秒")
        print(f"✅ 成功：{success_count} | ❌ 失败：{failed_count}")
        print(f"📈 成功率：{(success_count / total_rows * 100):.2f}%")

    return all_results


def generate_statistics(results: List[Dict]) -> Dict:
    """
    生成完全贴合你的数据集的统计报告
    """
    if not results:
        return {}

    total = len(results)
    success_results = [r for r in results if r["status"] == "success"]
    success_total = len(success_results)

    # 数量统计
    imaging_counts = [r["imaging_params_count"] for r in success_results]
    scattering_counts = [r["scattering_features_count"] for r in success_results]
    basic_counts = [r["basic_concepts_count"] for r in success_results]
    total_counts = [r["total_sar_terms_count"] for r in success_results]

    # 汇总统计
    total_imaging = sum(imaging_counts)
    total_scattering = sum(scattering_counts)
    total_basic = sum(basic_counts)
    total_terms = sum(total_counts)

    # 平均统计
    avg_imaging = np.mean(imaging_counts) if success_total > 0 else 0
    avg_scattering = np.mean(scattering_counts) if success_total > 0 else 0
    avg_basic = np.mean(basic_counts) if success_total > 0 else 0
    avg_total = np.mean(total_counts) if success_total > 0 else 0

    # 存在性统计
    has_imaging = sum(1 for r in success_results if r["imaging_params_count"] > 0)
    has_scattering = sum(1 for r in success_results if r["scattering_features_count"] > 0)
    has_basic = sum(1 for r in success_results if r["basic_concepts_count"] > 0)
    has_any = sum(1 for r in success_results if r["total_sar_terms_count"] > 0)

    # 分布统计
    total_dist = Counter(total_counts)

    statistics = {
        "一、基础处理统计": {
            "总处理行数": total,
            "成功处理行数": success_total,
            "失败处理行数": total - success_total,
            "处理成功率": f"{(success_total / total * 100):.2f}%"
        },
        "二、SAR术语总数量统计": {
            "成像参数总数量": total_imaging,
            "散射特性总数量": total_scattering,
            "基础概念总数量": total_basic,
            "SAR专业术语总数量": total_terms
        },
        "三、平均每个答案包含术语数": {
            "成像参数": f"{avg_imaging:.4f}",
            "散射特性": f"{avg_scattering:.4f}",
            "基础概念": f"{avg_basic:.4f}",
            "总计": f"{avg_total:.4f}"
        },
        "四、包含SAR术语的答案统计": {
            "包含成像参数的答案数": has_imaging,
            "占比": f"{(has_imaging / success_total * 100):.2f}%",
            "包含散射特性的答案数": has_scattering,
            "占比": f"{(has_scattering / success_total * 100):.2f}%",
            "包含基础概念的答案数": has_basic,
            "占比": f"{(has_basic / success_total * 100):.2f}%",
            "包含任意SAR术语的答案数": has_any,
            "占比": f"{(has_any / success_total * 100):.2f}%"
        },
        "五、术语数量分布（每个答案包含的术语数）": {
            f"{k}个术语": v for k, v in sorted(total_dist.items())
        }
    }

    return statistics


def save_results_and_statistics(results: List[Dict], statistics: Dict):
    """
    保存结果和统计报告
    """
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 保存详细结果
    result_path = os.path.join(RESULT_DIR, RESULT_FILENAME)
    try:
        with open(result_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            for res in results:
                filtered_res = {k: v for k, v in res.items() if k in OUTPUT_FIELDS}
                writer.writerow(filtered_res)
        with LOCK:
            print(f"\n💾 详细结果：{result_path}")
    except Exception as e:
        with LOCK:
            print(f"❌ 保存结果失败：{e}")

    # 保存统计报告
    stats_path = os.path.join(RESULT_DIR, STATISTICS_FILENAME)
    try:
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write("="*70 + "\n")
            f.write("SAR-VQA数据集专业术语统计报告\n")
            f.write("="*70 + "\n\n")

            for category, data in statistics.items():
                f.write(f"{category}\n")
                f.write("-"*40 + "\n")
                for key, value in data.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")

        with LOCK:
            print(f"💾 统计报告：{stats_path}")
    except Exception as e:
        with LOCK:
            print(f"❌ 保存报告失败：{e}")


def main():
    results = batch_process_answers()

    if not results:
        print("❌ 无处理结果")
        return

    statistics = generate_statistics(results)
    save_results_and_statistics(results, statistics)

    # 打印核心统计
    print("\n" + "="*70)
    print("📋 核心统计结果")
    print("="*70)
    print(f"\n总处理行数：{statistics['一、基础处理统计']['总处理行数']}")
    print(f"处理成功率：{statistics['一、基础处理统计']['处理成功率']}")
    print(f"\nSAR术语总数量：{statistics['二、SAR术语总数量统计']['SAR专业术语总数量']}")
    print(f"平均每个答案包含：{statistics['三、平均每个答案包含术语数']['总计']}个术语")
    print(f"\n包含SAR术语的答案占比：{statistics['四、包含SAR术语的答案统计']['占比']}")
    print(f"  - 成像参数：{statistics['四、包含SAR术语的答案统计']['占比']}")
    print(f"  - 散射特性：{statistics['四、包含SAR术语的答案统计']['占比']}")
    print(f"  - 基础概念：{statistics['四、包含SAR术语的答案统计']['占比']}")
    
    print("\n🎉 分析完成！")


if __name__ == "__main__":
    main()