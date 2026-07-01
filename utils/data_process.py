import re
import base64


# 图片转换为Base64编码工具函数
def image_to_base64(image_path):
    """将本地图片转换为 Base64 编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 文本归一化工具函数
def normalize_text(text: str) -> str:
    """
    文本归一化：去除标点、转小写、合并多余空格
    用于后续相似度计算
    """
    text = str(text or "")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.lower().split())
