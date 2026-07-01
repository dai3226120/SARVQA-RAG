"""
文本处理工具函数
字符清洗、分词、数值解析
"""

import re
import pandas as pd

# ====================== 特殊字符映射 ======================
SPECIAL_CHAR_REPLACE_MAP = {
    '\x00': '',
    '\r': '',
    '\n': ' ',
    '\u3000': ' ',
    '\xa0': ' ',
    '\u2028': ' ',
}


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
