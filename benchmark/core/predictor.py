"""
模型预测统一入口
整合并行数据处理流水线
"""

import os
import time
import datetime
import concurrent.futures

import pandas as pd

from config import data_config, path_config, prompt_config
from utils.print_utils import safe_print, format_elapsed_time
from utils.text_utils import tokenize_text, clean_special_chars
from utils.csv_utils import build_image_path, save_results_to_csv
from .metrics import retrieval_metrics, compute_information_metrics


# ====================== 并行数据处理流水线 ======================
def process_vqa_data(
    process_func,
    csv_input_path,
    result_dir,
    base_filename,
    image_base_path,
    max_rows=None,
    max_workers=None,
    start_row=None,
    required_columns=None,
    use_timestamp=True,
    batch_save_threshold=None,
):
    """
    通用的 SAR-VQA 数据并行处理函数

    参数:
        process_func: 处理单行数据的函数，签名为 (index, row, image_base_path) -> dict
        csv_input_path: 输入 CSV 路径
        result_dir: 结果目录
        base_filename: 结果文件前缀
        image_base_path: 图像基础路径
        max_rows: 最大处理行数
        max_workers: 最大并发数
        start_row: 起始行
        required_columns: 必需列名列表
    """
    safe_print("=" * 80)
    safe_print("[START] 开始处理 SAR-VQA 数据")
    safe_print("=" * 80)

    if max_rows is None:
        max_rows = data_config.MAX_PROCESS_ROWS
    if max_workers is None:
        max_workers = data_config.MAX_WORKERS
    if start_row is None:
        start_row = data_config.START_ROW
    if required_columns is None:
        required_columns = data_config.REQUIRED_COLUMNS
    if batch_save_threshold is None:
        batch_save_threshold = data_config.BATCH_SAVE_THRESHOLD

    safe_print("[CONFIG] 处理配置:")
    safe_print(f"  - 输入文件: {csv_input_path}")
    safe_print(f"  - 结果目录: {result_dir}")
    safe_print(f"  - 图像路径: {image_base_path}")
    safe_print(f"  - 最大行数: {max_rows}")
    safe_print(f"  - 起始行: {start_row}")
    safe_print(f"  - 最大并发: {max_workers}")
    safe_print(f"  - 批次阈值: {batch_save_threshold}")
    safe_print(f"  - 必需列: {required_columns}")
    safe_print("-" * 80)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if use_timestamp:
        output_csv_path = os.path.join(result_dir, f"{base_filename}_{timestamp}.csv")
    else:
        output_csv_path = os.path.join(result_dir, f"{base_filename}.csv")
    latest_csv_path = os.path.join(result_dir, f"{base_filename}_latest.csv")

    os.makedirs(result_dir, exist_ok=True)

    if not os.path.exists(csv_input_path):
        safe_print(f"[ERROR] CSV 文件不存在: {csv_input_path}")
        return None

    try:
        safe_print("[INFO] 正在读取 CSV 文件...")
        if start_row == 0:
            df = pd.read_csv(csv_input_path, nrows=max_rows, encoding=data_config.CSV_READ_ENCODING)
        else:
            df = pd.read_csv(
                csv_input_path,
                skiprows=range(1, start_row),
                nrows=max_rows,
                encoding=data_config.CSV_READ_ENCODING
            )
        df.columns = df.columns.str.strip()

        safe_print("[OK] 成功读取 CSV 文件")
        safe_print(f"   - 总行数: {len(df)}")
        safe_print(f"   - 列名: {df.columns.tolist()}")

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            safe_print(f"[ERROR] 缺少必要列: {missing_columns}")
            return None

    except Exception as e:
        safe_print(f"[ERROR] 读取 CSV 文件失败: {e}")
        return None

    results = []
    total_tasks = len(df)
    start_time = time.time()
    failed_count = 0

    safe_print(f"\n[PROCESS] 开始并行处理 ({total_tasks} 条任务)")
    safe_print(f"   线程数: {max_workers}")
    safe_print("-" * 80)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_row = {
            executor.submit(process_func, idx + start_row, row, image_base_path): (idx + start_row, row)
            for idx, row in df.iterrows()
        }

        for completed_count, future in enumerate(concurrent.futures.as_completed(future_to_row), start=1):
            idx, row = future_to_row[future]
            try:
                result_item = future.result()
                if result_item is not None:
                    results.append(result_item)

                if len(results) % batch_save_threshold == 0 or completed_count == total_tasks:
                    elapsed = time.time() - start_time
                    speed = completed_count / elapsed if elapsed > 0 else 0.0
                    progress = completed_count / total_tasks * 100 if total_tasks else 100.0
                    remaining = (total_tasks - completed_count) / speed if speed > 0 else 0

                    safe_print(f"\n[PROGRESS] 进度更新 [{completed_count}/{total_tasks}]")
                    safe_print(f"   进度: {progress:.1f}%")
                    safe_print(f"   速度: {speed:.2f} 条/秒")
                    safe_print(f"   已用时: {format_elapsed_time(elapsed)}")
                    safe_print(f"   预计剩余: {format_elapsed_time(remaining)}")
                    safe_print(f"   已完成: {len(results)}条有效记录 | 失败: {failed_count}条")

                    safe_print(f"\n[SAVE] 保存批次结果...")
                    save_results_to_csv(results, output_csv_path, latest_csv_path)
                    safe_print(f"   - {output_csv_path}")
                    safe_print(f"   - {latest_csv_path}")
                    safe_print("-" * 80)

            except Exception as e:
                failed_count += 1
                safe_print(f"[ERROR] 第{idx + 1}行出错: {str(e)}")

    try:
        result_df = save_results_to_csv(results, output_csv_path, latest_csv_path)

        elapsed = time.time() - start_time
        avg_per_input = elapsed / total_tasks if total_tasks else 0.0
        avg_per_output = elapsed / len(results) if results else 0.0
        success_rate = len(results) / total_tasks * 100 if total_tasks else 0

        safe_print("\n" + "=" * 80)
        safe_print("[DONE] 处理完成!")
        safe_print("=" * 80)
        safe_print("[STATS] 处理统计:")
        safe_print(f"   - 总任务数: {total_tasks}")
        safe_print(f"   - 成功: {len(results)}条")
        safe_print(f"   - 失败: {failed_count}条")
        safe_print(f"   - 成功率: {success_rate:.2f}%")
        safe_print("\n[TIME] 时间统计:")
        safe_print(f"   - 总时长: {format_elapsed_time(elapsed)}")
        safe_print(f"   - 平均耗时: {avg_per_input:.3f}秒/条")
        safe_print("\n[OUTPUT] 输出文件:")
        safe_print(f"   - {output_csv_path}")
        safe_print(f"   - {latest_csv_path}")
        safe_print("=" * 80)

        return result_df

    except Exception as e:
        safe_print(f"[ERROR] 保存失败: {e}")
        return None


def create_process_row_func(api_call_func, include_metrics=False):
    """
    创建处理单行数据的函数

    参数:
        api_call_func: API 调用函数，签名为 (image_path, question) -> str
        include_metrics: 是否包含指标计算
    """
    def process_single_row(index, row, image_base_path):
        try:
            id_text = clean_special_chars(row['id'])
            image_path = clean_special_chars(row['image'])
            question_text = clean_special_chars(row['question'])
            answer_text = clean_special_chars(row['answer'])

            image_path_front = build_image_path(image_path, image_base_path)

            predicted_answer = api_call_func(image_path_front, question_text)
            predicted_answer = clean_special_chars(predicted_answer)
            predicted_answer = predicted_answer.replace('\n', ' ').replace('\r', ' ').strip()

            result = {
                "id": id_text,
                "image": image_path,
                "question": question_text.lower(),
                "answer": answer_text.lower(),
                "predicted": predicted_answer.lower()
            }

            if include_metrics:
                formatted_prompt = prompt_config.DEFAULT_PROMPT.format(question=question_text)
                ig_value, id_value = compute_information_metrics(question_text, formatted_prompt)
                result["IG"] = ig_value
                result["ID"] = id_value
                retrieval_metrics.add_ig_id(ig_value, id_value)

            return result

        except KeyError as e:
            safe_print(f"[ERROR] 第{index + 1}行数据访问错误: 列{e}不存在")
            return None
        except Exception as e:
            safe_print(f"[ERROR] 第{index + 1}行处理出错: {str(e)}")
            return None

    return process_single_row
