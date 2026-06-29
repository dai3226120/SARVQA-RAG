"""
豆包Seed + Agent VQA 预测脚本 - 执行入口
"""

import os
import sys

# 设置控制台编码为UTF-8，支持emoji显示
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

from config import ModelType, get_file_tag, get_csv_input_path, get_result_dir, get_base_filename
from utils import check_csv_structure, clean_special_chars, build_image_path, safe_print
from agent_client_rscsv import doubao_agent_client
from metrics import retrieval_metrics, compute_information_metrics
from data_processor import process_vqa_data


def process_single_row(index, row, image_base_path):
    """处理单行数据（包含指标计算）"""
    try:
        id_text = clean_special_chars(row['id'])
        image_path = clean_special_chars(row['image'])
        question_text = clean_special_chars(row['question'])
        answer_text = clean_special_chars(row['answer'])

        image_path_front = build_image_path(image_path, image_base_path)

        # 调用Agent
        predicted_answer = doubao_agent_client.call(image_path_front, question_text)
        predicted_answer = clean_special_chars(predicted_answer)

        # 获取检索耗时
        retrieval_latency = doubao_agent_client.get_last_retrieval_latency()

        # 计算信息指标
        from config import prompt_config
        ig_value, id_value = compute_information_metrics(question_text, prompt_config.AGENT_PROMPT.format(question=question_text))

        return {
            "id": id_text,
            "image": image_path,
            "question": question_text.lower(),
            "answer": answer_text.lower(),
            "predicted": predicted_answer.lower(),
            "retrieval_latency": round(retrieval_latency, 4),
            "IG": ig_value,
            "ID": id_value
        }

    except KeyError as e:
        safe_print(f"[ERROR] 第{index + 1}行数据访问错误: 列{e}不存在")
        return None
    except Exception as e:
        safe_print(f"[ERROR] 第{index + 1}行处理出错: {str(e)}")
        return None


def main():
    print("=" * 80)
    print("[INFO] SAR-VQA 数据处理脚本（豆包Seed + Agent）")
    print("=" * 80)

    # 配置参数
    file_tag = get_file_tag(ModelType.AGENT_DOUBAO)+"_rscsv"
    dataset_tag = "test"
    # dataset_tag = "val"
    csv_input_path = get_csv_input_path(dataset_tag)
    result_dir = get_result_dir(file_tag)
    base_filename = get_base_filename(file_tag, dataset_tag)
    image_base_path = "C:\\dataset/SAR-TEXT"

    # 打印配置信息
    safe_print("[CONFIG] 当前配置:")
    safe_print(f"   - 模型类型: {ModelType.AGENT_DOUBAO}")
    safe_print(f"   - 文件标签: {file_tag}")
    safe_print(f"   - 数据集: {dataset_tag}")
    safe_print(f"   - 输入CSV: {csv_input_path}")
    safe_print(f"   - 结果目录: {result_dir}")
    safe_print(f"   - 图像路径: {image_base_path}")
    safe_print("-" * 80)

    # 前置检查
    safe_print("[CHECK] 前置检查...")
    csv_dir = os.path.dirname(csv_input_path)
    if not os.path.exists(csv_dir):
        safe_print(f"[ERROR] 目录不存在: {csv_dir}")
        return
    safe_print(f"[OK] 目录存在: {csv_dir}")

    if not os.path.exists(csv_input_path):
        safe_print(f"[ERROR] CSV不存在: {csv_input_path}")
        return
    safe_print(f"[OK] CSV文件存在: {csv_input_path}")

    if not check_csv_structure(csv_input_path):
        safe_print("[ERROR] CSV结构检查失败")
        return
    safe_print("[OK] CSV结构检查通过")
    safe_print("-" * 80)

    # 重置指标
    safe_print("[INFO] 重置检索指标...")
    retrieval_metrics.reset()
    safe_print("[OK] 指标已重置")
    safe_print("")

    # 执行处理
    process_vqa_data(
        process_func=process_single_row,
        csv_input_path=csv_input_path,
        result_dir=result_dir,
        base_filename=base_filename,
        image_base_path=image_base_path
    )

    # 输出Agent调用统计
    doubao_agent_client.print_stats()

    # 输出指标汇总
    print("\n[STATS] 检索与信息指标汇总:")
    for line in retrieval_metrics.format_summary(doubao_agent_client):
        print(f"  {line}")

    print("\n[DONE] 全部处理完成！")


if __name__ == "__main__":
    main()
