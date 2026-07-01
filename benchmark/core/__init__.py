"""
core 模块
统一导出预测、评估、分析、指标四类功能
"""

from .predictor import (
    process_vqa_data,
    create_process_row_func,
)

from .benchmarker import (
    Benchmarker,
    call_chat_api,
    process_single_row,
    calculate_advanced_metrics,
)

from .metrics import (
    RetrievalMetrics,
    retrieval_metrics,
    compute_information_metrics,
    calculate_cosine_similarity,
    calculate_rouge_l,
    calculate_bleu_scores,
    calculate_meteor_score,
    calculate_all_metrics,
)

from .analyzer import (
    ResultAnalyzer,
    calculate_conf_threshold,
    patch_plt_show_for_save,
)

from .excel_summary import (
    update_excel_summary,
)
