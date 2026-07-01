"""
统一配置管理模块
从 YAML 文件加载配置，提供 config_handler 模式的统一配置访问接口
"""

import os
from dataclasses import dataclass, field
from typing import List

# 确保项目根目录在 sys.path 中（bootstrap 阶段必须用 __file__ 推导，之后统一走 path_tool）
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.path_tool import get_project_root, get_abs_path

from utils.config_handler import model_conf, chroma_conf, prompts_conf, agent_conf, eval_conf
from utils.prompt_loader import load_agent_prompts, load_default_prompts


# ====================== Benchmark 模型配置 ======================
@dataclass
class DoubaoConfig:
    API_ENDPOINT: str = model_conf['doubao_seed_full_endpoint']
    API_KEY: str = model_conf['doubao_seed_api_key']
    MODEL_NAME: str = model_conf['doubao-seed-2-0-mini_model_name']
    TIMEOUT: int = model_conf['doubao_seed_timeout']
    TEMPERATURE: float = model_conf['doubao_seed_temperature']
    THINKING_MODE: str = model_conf['doubao_seed_thinking_mode']


# ====================== Benchmark 提示词配置 ======================
@dataclass
class PromptConfig:
    DEFAULT_PROMPT: str = load_default_prompts()
    AGENT_PROMPT: str = load_agent_prompts()


# ====================== Benchmark 数据处理配置 ======================
@dataclass
class DataConfig:
    REQUIRED_COLUMNS: list = field(default_factory=lambda: eval_conf['data']['required_columns'])
    CSV_READ_ENCODING: str = eval_conf['data']['csv_read_encoding']
    CSV_WRITE_ENCODING: str = eval_conf['data']['csv_write_encoding']


# ====================== Benchmark 文件路径配置 ======================

def _resolve_path(path: str) -> str:
    """将相对路径转为基于项目根目录的绝对路径"""
    return os.path.normpath(os.path.join(_PROJECT_ROOT, path)) if not os.path.isabs(path) else path


@dataclass
class PathConfig:
    DATASET_SPLIT_DIR: str = _resolve_path(eval_conf['paths']['dataset_split_dir'])
    RESULT_DIR: str = _resolve_path(eval_conf['paths']['result_dir'])
    BENCHMARK_RESULT_DIR: str = _resolve_path(eval_conf['paths']['benchmark_result_dir'])
    IMAGE_BASE_PATH: str = eval_conf['paths']['image_base_path']
    AGENT_ROOT_DIR: str = _PROJECT_ROOT


# ====================== Benchmark 分析配置 ======================
@dataclass
class AnalysisConfig:
    CONFIDENCE_THRESHOLD: float = eval_conf['analysis']['confidence_threshold']
    TARGET_METRICS: list = field(default_factory=lambda: eval_conf['analysis']['target_metrics'])


# ====================== 全局实例 ======================
data_config = DataConfig()
path_config = PathConfig()
prompt_config = PromptConfig()
doubao_config = DoubaoConfig()
analysis_config = AnalysisConfig()


# ====================== 模型类型枚举 ======================
class ModelType:
    DOUBAO = eval_conf['model_types']['doubao']
    INTERNVL = eval_conf['model_types']['internvl']
    AGENT_DOUBAO = eval_conf['model_types']['agent_doubao']
    AGENT_INTERNVL = eval_conf['model_types']['agent_internvl']


def get_file_tag(model_type: str) -> str:
    """根据模型类型获取文件标签"""
    return eval_conf['file_tags'].get(model_type, model_type)


def get_csv_input_path(dataset_tag: str = "val") -> str:
    """获取输入CSV路径"""
    return os.path.join(path_config.DATASET_SPLIT_DIR, f"{dataset_tag}.csv")


def get_result_dir(file_tag: str) -> str:
    """获取结果目录"""
    return os.path.join(path_config.RESULT_DIR, file_tag)


def get_base_filename(file_tag: str, dataset_tag: str) -> str:
    """获取基础文件名"""
    return f"{file_tag}_{dataset_tag}_predicted_question"
