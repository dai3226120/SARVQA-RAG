"""
InternVL模型 VQA 预测脚本 - 执行入口
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

from config import ModelType, get_file_tag, get_csv_input_path, get_result_dir, get_base_filename, internvl_config
from utils import check_csv_structure, safe_print
from api_client import call_internvl_api, internvl_client
from data_processor import process_vqa_data, create_process_row_func
from metrics import retrieval_metrics


def main():
    print("=" * 80)
    print("[START] SAR-VQA 数据处理脚本（InternVL）")
    print("=" * 80)

    # 配置参数
    file_tag = get_file_tag(ModelType.INTERNVL)
    dataset_tag = "val"
    csv_input_path = get_csv_input_path(dataset_tag)
    result_dir = get_result_dir(file_tag)
    base_filename = get_base_filename(file_tag, dataset_tag)
    image_base_path = "C:\\dataset/SAR-TEXT"

    # 打印配置信息
    safe_print("[CONFIG] 当前配置:")
    safe_print(f"   - 模型类型: {ModelType.INTERNVL}")
    safe_print(f"   - 文件标签: {file_tag}")
    safe_print(f"   - 数据集: {dataset_tag}")
    safe_print(f"   - 输入CSV: {csv_input_path}")
    safe_print(f"   - 结果目录: {result_dir}")
    safe_print(f"   - 图像路径: {image_base_path}")
    safe_print(f"   - API端点: {internvl_config.API_ENDPOINT}")
    safe_print(f"   - 模型名称: {internvl_config.MODEL_NAME}")
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
    safe_print("[INFO] 重置信息指标...")
    retrieval_metrics.reset()
    safe_print("[OK] 指标已重置")
    safe_print("")

    # 创建处理函数并执行
    safe_print("[INFO] 创建处理函数...")
    process_func = create_process_row_func(call_internvl_api, include_metrics=True)
    safe_print("[OK] 处理函数创建成功")
    safe_print("")

    # 执行处理
    process_vqa_data(
        process_func=process_func,
        csv_input_path=csv_input_path,
        result_dir=result_dir,
        base_filename=base_filename,
        image_base_path=image_base_path
    )

    # 输出API调用统计
    internvl_client.print_stats()

    # 输出指标汇总
    print("\n[STATS] 信息指标汇总:")
    for line in retrieval_metrics.format_summary():
        print(f"  {line}")

    safe_print("\n[DONE] 全部处理完成！")


if __name__ == "__main__":
    main()
