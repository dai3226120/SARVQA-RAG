"""
CSV 与路径处理工具函数
"""

import os
import pandas as pd


def build_image_path(image_path, image_base_path):
    """构建完整的图像路径"""
    if image_path.startswith('/'):
        return os.path.join(image_base_path, image_path.lstrip('/'))
    return os.path.join(image_base_path, image_path)


def ensure_dir_exists(dir_path):
    """确保目录存在"""
    os.makedirs(dir_path, exist_ok=True)


def check_csv_structure(csv_file_path, required_columns=None, csv_read_encoding='utf_8_sig'):
    """检查 CSV 文件的结构和列名"""
    try:
        df_sample = pd.read_csv(csv_file_path, nrows=5, encoding=csv_read_encoding)
        print("CSV 文件结构检查:")
        print(f"  列名: {df_sample.columns.tolist()}")
        print(f"  前 5 行数据预览:")
        for i, row in df_sample.iterrows():
            preview = {k: str(v)[:30] for k, v in row.to_dict().items()}
            print(f"    行{i + 1}: {preview}")
        return True
    except Exception as e:
        print(f"检查 CSV 文件结构失败: {e}")
        return False


def save_results_to_csv(results, output_csv_path, latest_csv_path=None, csv_write_encoding='utf_8_sig'):
    """保存结果到 CSV 文件"""
    if not results:
        return None

    ensure_dir_exists(os.path.dirname(output_csv_path))

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv_path, index=False, encoding=csv_write_encoding)

    if latest_csv_path:
        result_df.to_csv(latest_csv_path, index=False, encoding=csv_write_encoding)

    return result_df




