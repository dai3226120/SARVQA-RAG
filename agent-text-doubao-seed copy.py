import pandas as pd
import requests
import json
import os
import base64
from pathlib import Path
import datetime
import time
import concurrent.futures
import threading  # 用于线程安全的锁机制

# -------------------------- MainAgent 相关导入 --------------------------
from dotenv import load_dotenv
# load_dotenv()

from langchain_core.messages import HumanMessage
# 导入项目内的 MainAgent 相关模块（请根据实际路径调整）

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from agent.mainagent import MainAgent
# from agent.mainagent_rscsv import MainAgent
from agent.tools.middleware import calculate_hit_rate
# -----------------------------------------------------------------------------

# -------------------------- 全局可配置常量 --------------------------
# ====================== 线程安全配置 ======================
# 全局锁，保证打印和文件写入的线程安全
lock = threading.Lock()

# ====================== 统计指标配置 ======================
retrieval_metrics = {
    "retrieval_latencies_ms": [],
    "slice_retrieval_latencies_ms": [],
    "history_log_hits": 0,
    "history_log_attempts": 0,
    "information_gain_values": [],
    "information_density_values": [],
}

def _safe_parse_float(text):
    try:
        return float(text)
    except Exception:
        return None

def _percentile(values, percentile):
    if not values:
        return None
    values_sorted = sorted(values)
    rank = (len(values_sorted) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(values_sorted) - 1)
    weight = rank - lower
    return values_sorted[lower] * (1 - weight) + values_sorted[upper] * weight

def _tokenize_text(text):
    import re
    return re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", text.lower())

def _compute_information_metrics(question, prompt_text):
    stop_words = {
        'the','a','an','and','or','is','are','was','were','in','on','at','to','for','of','with','by','this','that','these','those','it','its',
        'as','from','into','about','if','then','but','also','be','being','been','can','could','would','should','may','might','will','shall',
        'has','have','had','do','does','did','not','no','so','such','than','too','very','more','most','some','any','all','their','there',
        'when','where','who','what','which','while','during','among','were','each','other','per','through','over','under','between','without',
        'within','对于','和','与','与','也','有','是','在','的','了','不','或','及','且','这','那','其','被','为'
    }
    q_tokens = [t for t in _tokenize_text(question) if t and t not in stop_words]
    p_tokens = [t for t in _tokenize_text(prompt_text) if t and t not in stop_words]
    raw_prompt_tokens = _tokenize_text(prompt_text)
    ig = 0.0
    if len(q_tokens) > 0:
        ig = (len(p_tokens) - len(q_tokens)) / len(q_tokens)
    id_value = 0.0
    if len(raw_prompt_tokens) > 0:
        id_value = len(p_tokens) / len(raw_prompt_tokens)
    return round(ig, 4), round(id_value, 4)


def _update_retrieval_metrics(raw_text):
    import re
    if not raw_text:
        return
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    with lock:
        for line in lines:
            low_line = line.lower()
            is_membership_line = "隶属度命中" in line or "隶属度未命中" in line or "隶属度未达标" in line
            if any(key in low_line for key in ["历史日志", "history log", "rs_rscsv"]) or is_membership_line:
                if "隶属度命中" in line or "命中" in low_line or "hit" in low_line:
                    retrieval_metrics["history_log_hits"] += 1
                if "隶属度未命中" in line or "未命中" in low_line or "miss" in low_line or "未达标" in low_line:
                    pass
                retrieval_metrics["history_log_attempts"] += 1
            if any(keyword in low_line for keyword in ["检索", "retrieval", "slice", "切片", "rs_rscsv", "history", "hit"]):
                for match in re.finditer(r"([0-9]+(?:\.[0-9]+)?)\s*(ms|毫秒)", line):
                    latency = _safe_parse_float(match.group(1))
                    if latency is None:
                        continue
                    if "slice" in low_line or "切片" in low_line or "知识切片" in low_line:
                        retrieval_metrics["slice_retrieval_latencies_ms"].append(latency)
                    else:
                        retrieval_metrics["retrieval_latencies_ms"].append(latency)

def _format_metrics_summary():
    total_attempts = retrieval_metrics["history_log_attempts"]
    hit_rate = None
    if total_attempts > 0:
        hit_rate = retrieval_metrics["history_log_hits"] / total_attempts * 100.0
    avg_ig = None
    if retrieval_metrics["information_gain_values"]:
        avg_ig = sum(retrieval_metrics["information_gain_values"]) / len(retrieval_metrics["information_gain_values"])
    avg_id = None
    if retrieval_metrics["information_density_values"]:
        avg_id = sum(retrieval_metrics["information_density_values"]) / len(retrieval_metrics["information_density_values"])
    summary_lines = []
    summary_lines.append(f"隶属度命中率: {hit_rate:.2f}%" if hit_rate is not None else "隶属度命中率: N/A")
    summary_lines.append(f"信息增益度 IG: {avg_ig:.4f}" if avg_ig is not None else "信息增益度 IG: N/A")
    summary_lines.append(f"信息密度 ID: {avg_id:.4f}" if avg_id is not None else "信息密度 ID: N/A")
    return summary_lines

# ====================== 数据处理配置 ======================
MAX_PROCESS_ROWS = 200  # 最大处理行数
START_ROW = 0  # 新增：起始处理行数（从0开始计数，0代表第一行）
MAX_WORKERS = 50  # 线程池最大并发数
BATCH_SAVE_THRESHOLD = 100  # 每处理N条数据增量保存一次
REQUIRED_COLUMNS = ['id', 'image', 'question', 'answer']  # 必需列
CSV_READ_ENCODING = 'utf_8_sig'  # CSV读取编码
CSV_WRITE_ENCODING = 'utf_8_sig'  # CSV写入编码
SPECIAL_CHAR_REPLACE_MAP = {
    '\x00': '',
    '\r': '',
    '\n': ' ',
    '\u3000': ' ',
    '\xa0': ' ',
    '\u2028': ' ', 
}  # 需要清理的特殊字符映射

# ====================== 文件路径配置 ======================
FILE_TAG = "agent-text-doubao-seed-2-0-mini"
# DATASET_TAG = "test"
DATASET_TAG = "val"
CSV_INPUT_PATH = f"./dataset_split/{DATASET_TAG}.csv"
IMAGE_BASE_PATH = "C:\\dataset/SAR-TEXT"  # 图像文件基础路径
RESULT_DIR = f"./benchmark/result/{FILE_TAG}"
BASE_FILENAME = f"{FILE_TAG}_{DATASET_TAG}_predicted_question"  # 输出文件前缀

# ====================== 提示词配置 ======================
PROMPT_TEMPLATE = """
Answer the following question about the SAR image in a detailed, descriptive style. Your answer should be strictly limited to 150 words.
Your response should:
1. Be in complete sentences
2. Use formal, technical but clear language (like marine/Remote Sensing imaging terminology)
3. Provide specific details observable in the image
4. Maintain a consistent tone with professional Remote Sensing image analysis
5. Avoid short answers - be descriptive and comprehensive
Answer in English.
Question: {question}
""".strip()
# ---------------------------------------------------------------------------------

# ===================== 全局初始化 =====================
# 初始化 MainAgent（全局单例，避免重复初始化）
agent = None
def init_agent():
    """初始化 MainAgent 实例"""
    global agent
    if agent is None:
        with lock:
            print("🔧 初始化 MainAgent 实例...")
        agent = MainAgent()
    return agent

# ===================== 功能函数 =====================
def call_chat_api(image_path_front, question, *args, **kwargs):
    """
    替换为调用 MainAgent 进行预测（替代原API调用）
    新增：过滤工具调用日志，只保留纯回答内容
    """
    # 检查图像文件是否存在
    if not os.path.exists(image_path_front):
        with lock:  # 加锁避免打印混乱
            print(f"警告: 图像文件不存在: {image_path_front}")
        return "Image file not found"
    
    try:
        # 初始化 Agent
        init_agent()
        
        # 构建增强提示词
        formatted_prompt = PROMPT_TEMPLATE.format(question=question)
        
        # 读取图像文件并准备传递给 Agent
        with open(image_path_front, 'rb') as image_file:
            # 调用 Agent 的 execute_stream 方法获取结果
            response_chunks = []
            for chunk in agent.execute_stream(query=formatted_prompt, image_file=image_file):
                response_chunks.append(chunk)
            
            # 拼接完整响应
            full_response = ''.join(response_chunks).strip()
            try:
                ig_value, id_value = _compute_information_metrics(question, formatted_prompt)
                with lock:
                    retrieval_metrics["information_gain_values"].append(ig_value)
                    retrieval_metrics["information_density_values"].append(id_value)
            except Exception:
                pass
            try:
                _update_retrieval_metrics(full_response)
            except Exception:
                pass
        
        # ========== 过滤工具调用日志 ==========
        # 定义需要过滤的工具日志前缀（根据实际输出补充）
        FILTER_PREFIXES = [
            "rag_rscsv",
            "rag_summarize",
            "【匹配隶属度缓存】",
            "综合隶属度得分=",
            "q: ", "a: "  # 过滤问答对的前缀
        ]
        
        # 步骤1：按换行分割，逐行过滤
        filtered_lines = []
        for line in full_response.split('\n'):
            line = line.strip()
            # 跳过包含工具日志前缀的行
            if any(prefix in line.lower() for prefix in FILTER_PREFIXES):
                continue
            # 跳过空行
            if not line:
                continue
            filtered_lines.append(line)
        
        # 步骤2：拼接过滤后的纯回答
        pure_answer = ' '.join(filtered_lines).strip()
        
        # 步骤3：兜底清理（如果过滤后为空，尝试从原始响应中提取后半部分）
        if not pure_answer:
            # 备选方案：按中文标点/英文标点分割，取最后一段
            import re
            # 按【】、()、：等分隔符分割，取最后一部分
            parts = re.split(r'【|】|\(|\)|：|:', full_response)
            pure_answer = parts[-1].strip() if parts else full_response
        
        # ========== 原有清理逻辑 ==========
        final_answer = pure_answer
        # 清理多余格式
        final_answer = final_answer.replace("--- [Tool Output:", "").replace("] ---", "")
        final_answer = final_answer.replace("---------------------------", "")
        final_answer = ' '.join(final_answer.split()[:200])  # 限制长度

        return final_answer

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        with lock:
            print(f"调用Agent出错: {error_msg}")
        return error_msg

def clean_special_chars(text):
    """
    清理特殊字符，避免乱码
    """
    if pd.isna(text) or text is None:
        return ""
    # 转为字符串并清理特殊字符
    clean_text = str(text).strip()
    for char, replacement in SPECIAL_CHAR_REPLACE_MAP.items():
        clean_text = clean_text.replace(char, replacement)
    return clean_text

def process_single_row(index, row, image_base_path=IMAGE_BASE_PATH):
    """
    处理单行数据（供线程池调用）
    适配新数据集格式：id,image,question,answer
    返回：处理后的结果字典，失败返回None
    """
    try:
        # 安全获取列数据并清理特殊字符（修复乱码核心步骤1）
        id_text = clean_special_chars(row['id'])
        image_path = clean_special_chars(row['image'])
        question_text = clean_special_chars(row['question'])
        answer_text = clean_special_chars(row['answer'])
        
        # 构建完整图像路径
        # 处理image路径的绝对/相对路径问题
        if image_path.startswith('/'):
            # 如果是绝对路径，拼接base路径（去掉开头的/避免路径错误）
            image_path_front = os.path.join(image_base_path, image_path.lstrip('/'))
        else:
            image_path_front = os.path.join(image_base_path, image_path)

        # 获取预测结果（已过滤为纯答案）
        predicted_answer = call_chat_api(image_path_front, question_text)
        predicted_answer = clean_special_chars(predicted_answer)
        ig_value, id_value = _compute_information_metrics(question_text, PROMPT_TEMPLATE.format(question=question_text))
        
        # 构建结果项（包含所有要求的字段）
        result_item = {
            "id": id_text,
            "image": image_path,  # 保持原字段名
            "question": question_text.lower(),
            "answer": answer_text.lower(),
            "predicted": predicted_answer.lower(),  # 预测结果字段
            "IG": ig_value,
            "ID": id_value,
        }
        
        return result_item
        
    except KeyError as e:
        with lock:
            print(f"❌ 第{index + 1}行数据访问错误: 列{e}不存在")
        return None
    except Exception as e:
        with lock:
            print(f"❌ 处理第{index + 1}行数据时出错: {str(e)}")
        return None

def process_vqa_data_with_limit(max_rows=MAX_PROCESS_ROWS, max_workers=MAX_WORKERS, start_row=START_ROW):
    """
    处理SAR-VQA数据并生成CSV文件（并行版本）
    适配新数据集路径和格式，修复CSV读写乱码问题
    :param max_rows: 最大处理行数
    :param max_workers: 最大并发线程数（根据API限流调整）
    :param start_row: 起始处理行数（从0开始）
    """
    # 生成时间戳和文件路径
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 判断输入路径，生成对应的输出路径
    output_csv_path = os.path.join(RESULT_DIR, f"{BASE_FILENAME}_{timestamp}.csv")
    latest_csv_path = os.path.join(RESULT_DIR, f"{BASE_FILENAME}_latest.csv")
    
    # 确保结果目录存在
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # 检查CSV文件是否存在
    if not os.path.exists(CSV_INPUT_PATH):
        print(f"错误: CSV文件不存在: {CSV_INPUT_PATH}")
        return
    
    # 读取CSV文件（限制行数，指定编码避免读取乱码）
    try:
        # 先读取总行数（可选，用于校验起始行）
        if start_row == 0:
            # 从第0行开始，正常读取表头
            df = pd.read_csv(CSV_INPUT_PATH, nrows=max_rows, encoding=CSV_READ_ENCODING)
        else:
            # 从中间行开始：跳过数据，保留表头
            df = pd.read_csv(
                CSV_INPUT_PATH,
                skiprows=range(1, start_row),  # 只跳过数据行，不跳过表头
                nrows=max_rows,
                encoding=CSV_READ_ENCODING
            )
        df.columns = df.columns.str.strip()  # 清洗列名
        with lock:
            print(f"成功读取CSV文件的前{len(df)}行数据（限制: {max_rows}行）")
            print(f"CSV文件列名: {df.columns.tolist()}")
        
        # 检查必要列（适配新数据集）
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            print(f"错误: 缺少必要列: {missing_columns}")
            return
             
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return
    
    # 并行处理数据
    results = []
    total_tasks = len(df)
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务到线程池
        future_to_row = {
            # idx + start_row：还原为原始CSV的行号
            executor.submit(process_single_row, idx + start_row, row): (idx + start_row, row)
            for idx, row in df.iterrows()
        }
        
        # 遍历完成的任务，收集结果
        for completed_count, future in enumerate(concurrent.futures.as_completed(future_to_row), start=1):
            idx, row = future_to_row[future]
            try:
                result_item = future.result()
                if result_item is not None:
                    results.append(result_item)

                # 每批次保存时输出整体速率
                if len(results) % BATCH_SAVE_THRESHOLD == 0 or completed_count == total_tasks:
                    elapsed = time.time() - start_time
                    speed = completed_count / elapsed if elapsed > 0 else 0.0
                    progress = completed_count / total_tasks * 100 if total_tasks else 100.0
                    with lock:
                        print(f"⏱ 已处理 {completed_count}/{total_tasks} ({progress:.1f}%) | 平均速率 {speed:.2f} 条/秒 | 已用时 {elapsed:.1f} 秒")
                        print(f"📌 保存批次结果: {len(results)} 条数据")

                    with lock:
                        temp_df = pd.DataFrame(results)
                        temp_df.to_csv(output_csv_path, index=False, encoding=CSV_WRITE_ENCODING)
                        temp_df.to_csv(latest_csv_path, index=False, encoding=CSV_WRITE_ENCODING)
                        print(f"  - 时间戳文件: {output_csv_path}")
                        print(f"  - 最新文件: {latest_csv_path}")
            except Exception as e:
                with lock:
                    print(f"❌ 获取第{idx + 1}行结果时出错: {str(e)}")
    
    # 最终保存完整结果
    try:
        result_df = pd.DataFrame(results)
        # 最终保存也使用utf_8_sig编码
        result_df.to_csv(output_csv_path, index=False, encoding=CSV_WRITE_ENCODING)
        result_df.to_csv(latest_csv_path, index=False, encoding=CSV_WRITE_ENCODING)
        
        elapsed = time.time() - start_time
        avg_per_input = elapsed / total_tasks if total_tasks else 0.0
        avg_per_output = elapsed / len(results) if results else 0.0
        
        with lock:
            print(f"\n🎉 处理完成！总计{len(results)}条有效记录")
            print(f"⏱ 总时长: {elapsed:.1f} 秒")
            print(f"⚡ 平均每条（按输入行）: {avg_per_input:.3f} 秒/条")
            print(f"⚡ 平均每条（按有效输出）: {avg_per_output:.3f} 秒/条")
            print(f"📁 生成文件:")
            print(f"  - 时间戳文件: {output_csv_path}")
            print(f"  - 最新文件: {latest_csv_path}")
            print(f"📋 列名: {result_df.columns.tolist()}")
            for line in _format_metrics_summary():
                print(f"{line}")
            
    except Exception as e:
        print(f"❌ 保存CSV文件失败: {e}")

def check_csv_structure(csv_file_path=CSV_INPUT_PATH):
    """
    检查CSV文件的结构和列名，修复读取乱码问题
    """
    try:
        df_sample = pd.read_csv(csv_file_path, nrows=5, encoding=CSV_READ_ENCODING)
        print("📊 CSV文件结构检查:")
        print(f"列名: {df_sample.columns.tolist()}")
        print(f"前5行数据:\n{df_sample.head()}")
        return True
    except Exception as e:
        print(f"❌ 检查CSV文件结构失败: {e}")
        return False

def main():
    """
    主函数
    """
    print("🚀 开始并行处理SAR-VQA数据（使用MainAgent）...")
    
    # 前置检查
    csv_dir = os.path.dirname(CSV_INPUT_PATH)
    if not os.path.exists(csv_dir):
        print(f"❌ 错误: CSV文件目录不存在: {csv_dir}")
        return
    if not os.path.exists(CSV_INPUT_PATH):
        print(f"❌ 错误: CSV文件不存在: {CSV_INPUT_PATH}")
        return
    if not check_csv_structure():
        print("❌ CSV文件结构检查失败，请检查文件格式")
        return
    
    # 初始化Agent（提前初始化避免线程竞争）
    init_agent()
    
    # 并行处理
    process_vqa_data_with_limit(
        max_rows=MAX_PROCESS_ROWS,
        max_workers=MAX_WORKERS,
        start_row=START_ROW  # 传递起始行参数
    )
    
    hit_rates = calculate_hit_rate()
    print("\n=== 最终命中率统计 ===")
    for tool_name, rate in hit_rates.items():
        print(f"{tool_name}: {rate:.2%}")
        
    print("\n🏁 全部处理完成！")

if __name__ == "__main__":
    main()