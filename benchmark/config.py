"""
配置管理模块
统一管理所有配置参数
"""

import os
from dataclasses import dataclass


# ====================== 数据处理配置 ======================
@dataclass
class DataConfig:
    MAX_PROCESS_ROWS: int = 200000
    START_ROW: int = 0
    MAX_WORKERS: int = 30
    BATCH_SAVE_THRESHOLD: int = 100
    REQUIRED_COLUMNS: list = None
    CSV_READ_ENCODING: str = 'utf_8_sig'
    CSV_WRITE_ENCODING: str = 'utf_8_sig'
    
    def __post_init__(self):
        if self.REQUIRED_COLUMNS is None:
            self.REQUIRED_COLUMNS = ['id', 'image', 'question', 'answer']


# ====================== 文件路径配置 ======================
@dataclass
class PathConfig:
    DATASET_SPLIT_DIR: str = './dataset_split'
    RESULT_DIR: str = './benchmark/result'
    BENCHMARK_RESULT_DIR: str = './benchmark/result/benchmark_result'
    IMAGE_BASE_PATH: str = "C:\\dataset/SAR-TEXT"
    AGENT_ROOT_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ====================== 提示词配置 ======================
@dataclass
class PromptConfig:
    DEFAULT_PROMPT: str = """
Answer the following question about the SAR image in a detailed, descriptive style. Your answer should be strictly limited to 50 words.
Your response should:
1. Be in complete sentences
2. Use formal, technical but clear language (like marine/Remote Sensing imaging terminology)
3. Provide specific details observable in the image
4. Maintain a consistent tone with professional Remote Sensing image analysis
5. Avoid short answers - be descriptive and comprehensive

Question: {question}
""".strip()
    
    AGENT_PROMPT: str = """
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


# ====================== 豆包API配置 ======================
@dataclass
class DoubaoConfig:
    API_ENDPOINT: str = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    API_KEY: str = "e9f2dacd-d0a2-4c9a-ba2a-805ee0b40dcd"
    MODEL_NAME: str = "doubao-seed-2-0-mini-260428"
    TIMEOUT: int = 30
    TEMPERATURE: float = 0.7
    THINKING_MODE: str = "disabled"


# ====================== InternVL配置 ======================
@dataclass
class InternVLConfig:
    API_ENDPOINT: str = "http://i-2.gpushare.com:24667/v1/chat/completions"
    MODEL_NAME: str = "OpenGVLab/InternVL3_5-8B"
    TIMEOUT: int = 30
    TEMPERATURE: float = 0.2
    MAX_TOKENS: int = 1024


# ====================== 分析配置 ======================
@dataclass
class AnalysisConfig:
    CONFIDENCE_THRESHOLD: float = 0.85
    TARGET_METRICS: list = None
    
    def __post_init__(self):
        if self.TARGET_METRICS is None:
            self.TARGET_METRICS = ["cosine", "ROUGEL", "BLEU1", "BLEU2", "BLEU3", "BLEU4", "METEOR"]


# ====================== 全局实例 ======================
data_config = DataConfig()
path_config = PathConfig()
prompt_config = PromptConfig()
doubao_config = DoubaoConfig()
internvl_config = InternVLConfig()
analysis_config = AnalysisConfig()


# ====================== 模型类型枚举 ======================
class ModelType:
    DOUBAO = "doubao"
    INTERNVL = "internvl"
    AGENT_DOUBAO = "agent-doubao"
    AGENT_INTERNVL = "agent-internvl"


def get_file_tag(model_type: str) -> str:
    """根据模型类型获取文件标签"""
    tag_map = {
        ModelType.DOUBAO: "doubao-seed-2-0-mini-260428",
        ModelType.INTERNVL: "internvl3_5-8b",
        ModelType.AGENT_DOUBAO: "agent-text-doubao-seed-2-0-mini",
        ModelType.AGENT_INTERNVL: "agent-text-internVL3_5-8b",
    }
    return tag_map.get(model_type, model_type)


def get_csv_input_path(dataset_tag: str = "val") -> str:
    """获取输入CSV路径"""
    return os.path.join(path_config.DATASET_SPLIT_DIR, f"{dataset_tag}.csv")


def get_result_dir(file_tag: str) -> str:
    """获取结果目录"""
    return os.path.join(path_config.RESULT_DIR, file_tag)


def get_base_filename(file_tag: str, dataset_tag: str) -> str:
    """获取基础文件名"""
    return f"{file_tag}_{dataset_tag}_predicted_question"
