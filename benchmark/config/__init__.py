"""
配置管理模块
从 YAML 文件加载配置，提供统一配置访问接口
"""

import os
import sys
import yaml
from dataclasses import dataclass, field
from typing import List


_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# 确保项目根目录在 sys.path 中，使 benchmark 代码可以通过
# from agent.utils.xxx import ... 访问 agent 工具包
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_CONFIG_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _load_yaml(filename: str) -> dict:
    """加载 YAML 配置文件"""
    filepath = os.path.join(_CONFIG_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ====================== 加载 YAML 配置 ======================
_model_cfg = _load_yaml('model_config.yaml')
_agent_cfg = _load_yaml('agent_config.yaml')
_eval_cfg = _load_yaml('eval_config.yaml')


# ====================== 模型配置 ======================
@dataclass
class DoubaoConfig:
    API_ENDPOINT: str = _model_cfg['doubao_seed']['api_endpoint']
    API_KEY: str = _model_cfg['doubao_seed']['api_key']
    MODEL_NAME: str = _model_cfg['doubao_seed']['model_name']
    TIMEOUT: int = _model_cfg['doubao_seed']['timeout']
    TEMPERATURE: float = _model_cfg['doubao_seed']['temperature']
    THINKING_MODE: str = _model_cfg['doubao_seed']['thinking_mode']


@dataclass
class InternVLConfig:
    API_ENDPOINT: str = _model_cfg['internVL']['api_endpoint']
    MODEL_NAME: str = _model_cfg['internVL']['model_name']
    TIMEOUT: int = _model_cfg['internVL']['timeout']
    TEMPERATURE: float = _model_cfg['internVL']['temperature']
    MAX_TOKENS: int = _model_cfg['internVL']['max_tokens']


# ====================== 提示词配置 ======================
@dataclass
class PromptConfig:
    DEFAULT_PROMPT: str = _model_cfg['prompts']['default']
    AGENT_PROMPT: str = _agent_cfg['prompts']['agent']


# ====================== 数据处理配置 ======================
@dataclass
class DataConfig:
    MAX_PROCESS_ROWS: int = _eval_cfg['data']['max_process_rows']
    START_ROW: int = _eval_cfg['data']['start_row']
    MAX_WORKERS: int = _eval_cfg['data']['max_workers']
    BATCH_SAVE_THRESHOLD: int = _eval_cfg['data']['batch_save_threshold']
    REQUIRED_COLUMNS: list = field(default_factory=lambda: _eval_cfg['data']['required_columns'])
    CSV_READ_ENCODING: str = _eval_cfg['data']['csv_read_encoding']
    CSV_WRITE_ENCODING: str = _eval_cfg['data']['csv_write_encoding']


# ====================== 文件路径配置 ======================
@dataclass
class PathConfig:
    DATASET_SPLIT_DIR: str = _eval_cfg['paths']['dataset_split_dir']
    RESULT_DIR: str = _eval_cfg['paths']['result_dir']
    BENCHMARK_RESULT_DIR: str = _eval_cfg['paths']['benchmark_result_dir']
    IMAGE_BASE_PATH: str = _eval_cfg['paths']['image_base_path']
    AGENT_ROOT_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ====================== 分析配置 ======================
@dataclass
class AnalysisConfig:
    CONFIDENCE_THRESHOLD: float = _eval_cfg['analysis']['confidence_threshold']
    TARGET_METRICS: list = field(default_factory=lambda: _eval_cfg['analysis']['target_metrics'])


# ====================== 全局实例 ======================
data_config = DataConfig()
path_config = PathConfig()
prompt_config = PromptConfig()
doubao_config = DoubaoConfig()
internvl_config = InternVLConfig()
analysis_config = AnalysisConfig()


# ====================== 模型类型枚举 ======================
class ModelType:
    DOUBAO = _eval_cfg['model_types']['doubao']
    INTERNVL = _eval_cfg['model_types']['internvl']
    AGENT_DOUBAO = _eval_cfg['model_types']['agent_doubao']
    AGENT_INTERNVL = _eval_cfg['model_types']['agent_internvl']


def get_file_tag(model_type: str) -> str:
    """根据模型类型获取文件标签"""
    return _eval_cfg['file_tags'].get(model_type, model_type)


def get_csv_input_path(dataset_tag: str = "val") -> str:
    """获取输入CSV路径"""
    return os.path.join(path_config.DATASET_SPLIT_DIR, f"{dataset_tag}.csv")


def get_result_dir(file_tag: str) -> str:
    """获取结果目录"""
    return os.path.join(path_config.RESULT_DIR, file_tag)


def get_base_filename(file_tag: str, dataset_tag: str) -> str:
    """获取基础文件名"""
    return f"{file_tag}_{dataset_tag}_predicted_question"
