"""
通用工具函数模块
"""

import os
import re
import threading

import pandas as pd

from config import data_config


# ====================== 线程安全配置 ======================
lock = threading.Lock()


# ====================== 特殊字符映射 ======================
SPECIAL_CHAR_REPLACE_MAP = {
    '\x00': '',
    '\r': '',
    '\n': ' ',
    '\u3000': ' ',
    '\xa0': ' ',
    '\u2028': ' ',
}


# ====================== 文本处理函数 ======================
def clean_special_chars(text):
    """清理特殊字符，避免乱码"""
    if pd.isna(text) or text is None:
        return ""
    clean_text = str(text).strip()
    for char, replacement in SPECIAL_CHAR_REPLACE_MAP.items():
        clean_text = clean_text.replace(char, replacement)
    return clean_text


def tokenize_text(text):
    """分词处理，支持中英文"""
    return re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", text.lower())


def safe_parse_float(text):
    """安全解析浮点数"""
    try:
        return float(text)
    except Exception:
        return None


# ====================== 路径处理函数 ======================
def build_image_path(image_path, image_base_path):
    """构建完整的图像路径"""
    if image_path.startswith('/'):
        return os.path.join(image_base_path, image_path.lstrip('/'))
    return os.path.join(image_base_path, image_path)


def ensure_dir_exists(dir_path):
    """确保目录存在"""
    os.makedirs(dir_path, exist_ok=True)


# ====================== CSV处理函数 ======================
def check_csv_structure(csv_file_path, required_columns=None):
    """
    检查CSV文件的结构和列名
    """
    if required_columns is None:
        required_columns = data_config.REQUIRED_COLUMNS

    try:
        df_sample = pd.read_csv(csv_file_path, nrows=5, encoding=data_config.CSV_READ_ENCODING)
        print("📊 CSV文件结构检查:")
        print(f"  列名: {df_sample.columns.tolist()}")
        print(f"  前5行数据预览:")
        for i, row in df_sample.iterrows():
            preview = {k: str(v)[:30] for k, v in row.to_dict().items()}
            print(f"    行{i+1}: {preview}")
        return True
    except Exception as e:
        print(f"❌ 检查CSV文件结构失败: {e}")
        return False


def save_results_to_csv(results, output_csv_path, latest_csv_path=None):
    """保存结果到CSV文件"""
    if not results:
        return None

    ensure_dir_exists(os.path.dirname(output_csv_path))

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv_path, index=False, encoding=data_config.CSV_WRITE_ENCODING)

    if latest_csv_path:
        result_df.to_csv(latest_csv_path, index=False, encoding=data_config.CSV_WRITE_ENCODING)

    return result_df


# ====================== 数据准备函数 ======================
def prepare_row_data(row, image_base_path):
    """
    准备行数据，返回元组 (id, image_path, image_path_full, question, answer)
    """
    id_text = clean_special_chars(row['id'])
    image_path = clean_special_chars(row['image'])
    question_text = clean_special_chars(row['question'])
    answer_text = clean_special_chars(row['answer'])
    image_path_full = build_image_path(image_path, image_base_path)

    return id_text, image_path, image_path_full, question_text, answer_text


# ====================== 打印函数 ======================
def safe_print(*args, **kwargs):
    """线程安全的打印函数"""
    with lock:
        print(*args, **kwargs)


def format_elapsed_time(seconds):
    """格式化耗时"""
    if seconds < 0:
        return "计算中..."
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours}时{minutes}分{secs:.1f}秒"
    if minutes > 0:
        return f"{minutes}分{secs:.1f}秒"
    return f"{secs:.2f}秒"


def print_separator(title=None, char="="):
    """打印分隔线"""
    if title:
        safe_print(f"{char} {title} {char}")
        safe_print(char * 80)
    else:
        safe_print(char * 80)


def print_status(message, status="info"):
    """打印状态信息"""
    status_map = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "debug": "🔧"
    }
    safe_print(f"{status_map.get(status, 'ℹ️')} {message}")


def print_section(title):
    """打印章节标题"""
    safe_print("")
    safe_print("-" * 80)
    safe_print(f"📌 {title}")
    safe_print("-" * 80)
