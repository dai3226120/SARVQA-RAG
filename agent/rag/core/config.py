"""
RAG 子系统统一配置管理
将原散落在 rag_rsfit_builder.py Config 类 + chroma.yml 中的 RAG 相关参数集中管理
"""
from dataclasses import dataclass, field
from utils.config_handler import chroma_conf
from utils.path_tool import get_abs_path


@dataclass
class RagConfig:
    """RAG 子系统配置，集中管理路径与参数"""

    # ── 通用知识库 ──
    collection_name: str = chroma_conf.get("collection_name", "agent")
    persist_directory: str = get_abs_path(chroma_conf.get("persist_diretory", "agent/chroma_db"))
    data_path: str = get_abs_path(chroma_conf.get("data_path", "agent/data"))

    # ── 检索参数 ──
    k: int = int(chroma_conf.get("k", 1))
    top_p: int = int(chroma_conf.get("top_p", 9))
    slice_k: int = int(chroma_conf.get("slice_k", 50))
    membership_k: int = int(chroma_conf.get("membership_k", 50))

    # ── 隶属度权重 ──
    w1: float = float(chroma_conf.get("w1", 0.9))
    w2: float = float(chroma_conf.get("w2", 0.1))
    bleu_weight: float = float(chroma_conf.get("bleu_weight", 0.75))
    overlap_weight: float = float(chroma_conf.get("overlap_weight", 0.25))

    # ── 切分参数 ──
    chunk_size: int = int(chroma_conf.get("chunk_size", 500))
    chunk_overlap: int = int(chroma_conf.get("chunk_overlap", 50))
    separators: list = field(default_factory=lambda: chroma_conf.get("separators", ["\n\n", "\n", " ", ""]))

    # ── 聚类参数 ──
    n_clusters: int = int(chroma_conf.get("clustering", {}).get("n_clusters", 50))
    slice_size: int = int(chroma_conf.get("clustering", {}).get("slice_size", 5))

    # ── 贴合度参数 ──
    fit_threshold: float = float(chroma_conf.get("retrieval", {}).get("fit_threshold", 0.75))
    enable_rag_context: bool = bool(chroma_conf.get("retrieval", {}).get("enable_rag_context", True))

    # ── 向量库集合名 ──
    slices_collection_name: str = chroma_conf.get("collections", {}).get("slices", "sar_slices_collection")

    # ── MD5 记录文件 ──
    md5_store_path: str = get_abs_path(chroma_conf.get("md5_hex_store", "agent/md5.text"))
    md5_log_store_path: str = get_abs_path(chroma_conf.get("md5_log_store", "agent/md5_log.text"))

    # ── 知识库文件类型 ──
    allow_knowledge_file_type: list = field(
        default_factory=lambda: chroma_conf.get("allow_knowledge_file_type", ["txt", "pdf", "md"])
    )

    # ── 日志向量库 ──
    log_collection_name: str = "rag_test_logs"

    # ── VLM Agent 类型 ──
    vlm_agent_type: str = chroma_conf.get("vlm_agent_type", "doubao")

    # ── 测试配置 ──
    test_nrows: int = 50000
    base_image_dir: str = r"C:\dataset\SAR-TEXT"

    # ── 日志文件 ──
    log_name: str = "rag_feedback_logs.csv"
    error_log_name: str = "rag_error_logs.csv"

    # ── SAR 数据路径 ──
    sar_csv_path: str = get_abs_path("agent/data/SAR-VQA-180375.csv")

    @property
    def slice_csv_path(self) -> str:
        return get_abs_path(f"{self.data_path}/sar_slices.csv")

    @property
    def log_path(self) -> str:
        return get_abs_path(f"{self.data_path}/{self.log_name}")

    @property
    def pending_log_path(self) -> str:
        return get_abs_path(f"{self.data_path}/pending_records.csv")

    @property
    def error_log_path(self) -> str:
        return get_abs_path(f"{self.data_path}/{self.error_log_name}")


# 全局配置实例
rag_config = RagConfig()
