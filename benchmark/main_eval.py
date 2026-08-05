"""
SAR-VQA 主评估脚本 - 统一调用入口
通过 config / core / models 三个包实现预测→评估→分析流水线
"""

import os
import sys
import datetime

# 确保项目根目录在 sys.path 中（bootstrap 阶段必须用 __file__ 推导，之后统一走 path_tool）
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.path_tool import get_project_root, get_abs_path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# ---- 新模块导入 ----
import config as cfg
from core import (
    process_vqa_data,
    create_process_row_func,
    Benchmarker,
    ResultAnalyzer,
    patch_plt_show_for_save,
    update_excel_summary,
)
from models import (
    doubao_client,
    internvl_client,
    doubao_agent_client,
    internvl_agent_client,
    doubao_agent_rscsv_client,
    internvl_agent_rscsv_client,
    qwen37_plus_client,
)
from utils.print_utils import print_separator


# ====================== 模型映射配置 ======================
# 每个 MODEL_KEY 对应：客户端实例 / 调用函数 / 文件标签
# 模型类型：
#     - doubao-seed: 普通Doubao模型
#     - internVL: 普通InternVL模型
#     - agent-text-doubao-seed: 文本Doubao模型
#     - agent-text-doubao-seed_rscsv: 文本Doubao模型（RSCSV）
#     - agent-text-internVL: 文本InternVL模型
#     - agent-text-internVL_rscsv: 文本InternVL模型（RSCSV）

# MODEL_KEY = "doubao-seed"
# MODEL_KEY = "agent-text-doubao-seed"
# MODEL_KEY = "agent-text-doubao-seed_rscsv"

# MODEL_KEY = "internVL"
# MODEL_KEY = "agent-text-internVL"
# MODEL_KEY = "agent-text-internVL_rscsv"

MODEL_KEY = "qwen37-plus"


# ====================== 流水线模式配置 ======================
# 可选模式:
#   "full"   : 预测 + 评估 + 分析（完整流水线，结束后更新 Excel 统计表）
#   "eval"   : 评估 + 分析（跳过预测，需引用之前的预测结果）
#   "analyze": 仅分析（跳过预测与评估，需引用之前的评估结果）
PIPELINE_MODE = "full"
RESULT_TIMESTAMP = ""  # eval/analyze 必填：引用 RESULT_DIR/<模型类型>/<文件标签>/<时间戳>/ 下的历史结果

_PIPELINE_MODES = ("full", "eval", "analyze")


_MODEL_REGISTRY = {
    "doubao-seed": {
        "model_type": cfg.ModelType.DOUBAO,
        "file_tag": cfg.get_file_tag(cfg.ModelType.DOUBAO),
        "client": doubao_client,
        "api_call": doubao_client.call,
        "is_agent": False,
    },
    "internVL": {
        "model_type": cfg.ModelType.INTERNVL,
        "file_tag": cfg.get_file_tag(cfg.ModelType.INTERNVL),
        "client": internvl_client,
        "api_call": internvl_client.call,
        "is_agent": False,
    },
    "agent-text-doubao-seed": {
        "model_type": cfg.ModelType.AGENT_DOUBAO,
        "file_tag": cfg.get_file_tag(cfg.ModelType.AGENT_DOUBAO),
        "client": doubao_agent_client,
        "api_call": doubao_agent_client.call,
        "is_agent": True,
    },
    "agent-text-doubao-seed_rscsv": {
        "model_type": cfg.ModelType.AGENT_DOUBAO,
        "file_tag": "agent-text-doubao-seed-2-0-mini_rscsv",
        "client": doubao_agent_rscsv_client,
        "api_call": doubao_agent_rscsv_client.call,
        "is_agent": True,
    },
    "agent-text-internVL": {
        "model_type": cfg.ModelType.AGENT_INTERNVL,
        "file_tag": cfg.get_file_tag(cfg.ModelType.AGENT_INTERNVL),
        "client": internvl_agent_client,
        "api_call": internvl_agent_client.call,
        "is_agent": True,
    },
    "agent-text-internVL_rscsv": {
        "model_type": cfg.ModelType.AGENT_INTERNVL,
        "file_tag": "agent-text-internvl3-5-8b_rscsv",
        "client": internvl_agent_rscsv_client,
        "api_call": internvl_agent_rscsv_client.call,
        "is_agent": True,
    },
    "qwen37-plus": {
        "model_type": cfg.ModelType.QWEN37_PLUS,
        "file_tag": cfg.get_file_tag(cfg.ModelType.QWEN37_PLUS),
        "client": qwen37_plus_client,
        "api_call": qwen37_plus_client.call,
        "is_agent": False,
    },
    
}

# ====================== 数据集与路径配置 ======================
DATASET_TAG = "val"
IMAGE_BASE_PATH = cfg.path_config.IMAGE_BASE_PATH

# ====================== 数据处理参数（可在此处直接修改）======================
MAX_PROCESS_ROWS = 5000
START_ROW = 0
MAX_WORKERS = 100
BATCH_SAVE_THRESHOLD = 300

# ====================== 分析参数 ======================
CONFIDENCE_THRESHOLD = cfg.analysis_config.CONFIDENCE_THRESHOLD
TARGET_METRICS = cfg.analysis_config.TARGET_METRICS


# ====================== 流水线模式解析与校验 ======================
def _print_available_timestamps(base_dir: str):
    """打印 base_dir 下可用的历史运行时间戳目录（供用户修正 RESULT_TIMESTAMP）"""
    if not os.path.isdir(base_dir):
        print(f"   目录不存在: {base_dir}")
        return
    timestamps = sorted(
        (d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))),
        reverse=True,
    )
    if timestamps:
        print(f"   可用历史时间戳: {timestamps}")
    else:
        print(f"   目录 {base_dir} 下没有历史运行目录")


def resolve_pipeline_mode(model_type: str, file_tag: str):
    """校验并解析流水线模式配置。

    返回 (mode, ref_file):
        - mode 为 "full" / "eval" / "analyze"
        - ref_file 为 eval/analyze 模式下引用的历史结果文件，full 模式下为 None
    校验失败时打印错误信息并返回 None（调用方应终止流程）。
    """
    if PIPELINE_MODE not in _PIPELINE_MODES:
        print(f"❌ 无效的 PIPELINE_MODE: {PIPELINE_MODE}（可选: {' / '.join(_PIPELINE_MODES)}）")
        return None

    if PIPELINE_MODE == "full":
        return ("full", None)

    # eval / analyze 模式：解析引用的历史结果
    base_dir = os.path.join(cfg.path_config.RESULT_DIR, model_type, file_tag)
    if not RESULT_TIMESTAMP:
        print(f"❌ {PIPELINE_MODE} 模式必须指定 RESULT_TIMESTAMP（引用之前某次运行的结果）")
        _print_available_timestamps(base_dir)
        return None

    ref_dir = os.path.join(base_dir, RESULT_TIMESTAMP)
    if not os.path.isdir(ref_dir):
        print(f"❌ 引用目录不存在: {ref_dir}")
        _print_available_timestamps(base_dir)
        return None

    if PIPELINE_MODE == "eval":
        ref_file = os.path.join(ref_dir, f"{file_tag}_{DATASET_TAG}_predicted_question_latest.csv")
    else:  # analyze
        ref_file = os.path.join(ref_dir, f"{file_tag}_{DATASET_TAG}_benchmark_latest.csv")

    if not os.path.isfile(ref_file):
        print(f"❌ 引用文件不存在: {ref_file}")
        files = os.listdir(ref_dir)
        if files:
            print(f"   目录内可用文件: {files}")
        return None

    return (PIPELINE_MODE, ref_file)


# ====================== 步骤 1：模型预测 ======================
def run_prediction(model_key: str, output_dir: str) -> dict:
    """使用 core.predictor 执行模型预测，返回状态字典"""
    entry = _MODEL_REGISTRY.get(model_key)
    if not entry:
        print(f"❌ 未知模型: {model_key}")
        return {"success": False, "output_file": None, "stats": {}}

    file_tag = entry["file_tag"]
    api_call = entry["api_call"]
    client = entry["client"]
    is_agent = entry.get("is_agent", False)

    print_separator(f"步骤 1/3: 模型预测 ({model_key})")

    csv_input_path = cfg.get_csv_input_path(DATASET_TAG)
    base_filename = cfg.get_base_filename(file_tag, DATASET_TAG)

    print(f"\n📋 预测配置:")
    print(f"   - 模型: {model_key}")
    print(f"   - 文件标签: {file_tag}")
    print(f"   - 输入 CSV: {csv_input_path}")
    print(f"   - 输出目录: {output_dir}")
    print(f"   - 图像路径: {IMAGE_BASE_PATH}")
    print(f"   - 最大行数: {MAX_PROCESS_ROWS} | 起始行: {START_ROW}")
    print(f"   - 最大并发: {MAX_WORKERS} | 批次阈值: {BATCH_SAVE_THRESHOLD}")

    # Agent 模型使用 AGENT_PROMPT 计算 IG/ID，并传入 client 以获取 RAG 增强的 IG/ID
    if is_agent:
        process_func = create_process_row_func(
            api_call,
            include_metrics=True,
            prompt_template=cfg.prompt_config.AGENT_PROMPT,
            agent_client=client,
        )
    else:
        process_func = create_process_row_func(
            api_call,
            include_metrics=True,
            prompt_template=cfg.prompt_config.DEFAULT_PROMPT,
        )

    print_separator(char="-")

    try:
        result_df = process_vqa_data(
            process_func=process_func,
            csv_input_path=csv_input_path,
            result_dir=output_dir,
            base_filename=base_filename,
            image_base_path=IMAGE_BASE_PATH,
            use_timestamp=False,
            max_rows=MAX_PROCESS_ROWS,
            max_workers=MAX_WORKERS,
            start_row=START_ROW,
            batch_save_threshold=BATCH_SAVE_THRESHOLD,
        )

        if result_df is not None:
            if hasattr(client, 'print_stats'):
                client.print_stats()
            client_stats = client.get_stats() if hasattr(client, 'get_stats') else {}
            retrieval_stats = client.get_retrieval_latency_stats() if hasattr(client, 'get_retrieval_latency_stats') else {}
            membership_hit_rate = client.get_rag_rscsv_membership_hit_rate() if hasattr(client, 'get_rag_rscsv_membership_hit_rate') else None
            output_file = os.path.join(output_dir, f"{base_filename}_latest.csv")
            print(f"✅ 模型预测完成")
            return {
                "success": True,
                "output_file": output_file,
                "stats": {
                    "total_rows": len(result_df),
                    **client_stats,
                },
                "retrieval_stats": retrieval_stats,
                "membership_hit_rate": membership_hit_rate,
            }
        else:
            print(f"❌ 模型预测返回空结果")
            return {"success": False, "output_file": None, "stats": {}}

    except Exception as e:
        print(f"❌ 模型预测失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "output_file": None, "stats": {}}


# ====================== 步骤 2：指标评估 ======================
def run_benchmark(file_tag: str, output_dir: str, predicted_csv: str = None) -> dict:
    """使用 core.benchmarker.Benchmarker 执行指标评估，返回状态字典"""
    print_separator(f"步骤 2/3: 指标评估 (FILE_TAG={file_tag})")

    if predicted_csv is None:
        predicted_csv = os.path.join(output_dir, f"{file_tag}_{DATASET_TAG}_predicted_question_latest.csv")

    base_filename = f"{file_tag}_{DATASET_TAG}_benchmark"

    print(f"\n📋 评估配置:")
    print(f"   - 输入: {predicted_csv}")
    print(f"   - 输出目录: {output_dir}")
    print(f"   - 基础文件名: {base_filename}")

    print_separator(char="-")

    try:
        benchmarker = Benchmarker(
            input_csv_path=predicted_csv,
            result_dir=output_dir,
            base_filename=base_filename,
        )

        if not benchmarker.check_input_csv_structure():
            print("❌ 输入 CSV 结构检查失败")
            return {"success": False, "output_file": None, "stats": {}}

        output_path = benchmarker.run(use_timestamp=False)
        if output_path:
            stats = {
                "total_rows": len(benchmarker.results),
                "success_count": benchmarker.success_count,
                "failed_count": benchmarker.failed_count,
                "total_time": benchmarker.total_time,
            }
            # 计算平均指标
            success_results = [r for r in benchmarker.results if r.get("status") == "success"]
            if success_results:
                import numpy as np
                stats["match_count"] = sum(1 for r in success_results if r.get("correct") in ("1", 1))
                stats["match_rate"] = stats["match_count"] / len(success_results) if success_results else 0
                stats["avg_cosine"] = float(np.mean([r["cosine"] for r in success_results]))
                stats["avg_rouge_l"] = float(np.mean([r["ROUGEL"] for r in success_results]))
                stats["avg_ig"] = float(np.mean([r.get("IG", 0) for r in success_results]))
                stats["avg_id"] = float(np.mean([r.get("ID", 0) for r in success_results]))
            print(f"✅ 指标评估完成 → {output_path}")
            return {"success": True, "output_file": output_path, "stats": stats}
        else:
            print(f"❌ 指标评估失败")
            return {"success": False, "output_file": None, "stats": {}}

    except Exception as e:
        print(f"❌ 指标评估失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "output_file": None, "stats": {}}


# ====================== 步骤 3：结果分析 ======================
def run_analysis(file_tag: str, output_dir: str, benchmark_csv: str = None) -> dict:
    """使用 core.analyzer.ResultAnalyzer 执行结果分析，返回状态字典"""
    print_separator(f"步骤 3/3: 结果分析 (FILE_TAG={file_tag})")

    plt_save_dir = os.path.join(output_dir, "plots")
    os.makedirs(plt_save_dir, exist_ok=True)
    print(f"📁 图表保存目录: {plt_save_dir}")

    if benchmark_csv is None:
        benchmark_csv = os.path.join(output_dir, f"{file_tag}_{DATASET_TAG}_benchmark_latest.csv")

    print(f"\n📋 分析配置:")
    print(f"   - 输入: {benchmark_csv}")
    print(f"   - 置信度阈值: {CONFIDENCE_THRESHOLD}")
    print(f"   - 目标指标: {TARGET_METRICS}")

    print_separator(char="-")

    # 使用 core.analyzer 提供的图表保存工具
    original_show, plot_counter = patch_plt_show_for_save(plt_save_dir, file_tag)

    try:
        analyzer = ResultAnalyzer(
            csv_path=benchmark_csv,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            target_metrics=TARGET_METRICS,
        )
        results_summary = analyzer.analyze(plot=True, save_dir=plt_save_dir)

        plt.show = original_show
        print(f"\n✅ 结果分析完成")
        print(f"📁 图表已保存至: {plt_save_dir}")
        return {
            "success": True,
            "plots_dir": plt_save_dir,
            "stats": {
                "plot_count": plot_counter[0],
                "avg_pred_tokens": analyzer.avg_pred_tokens,
            },
            "metrics_avg": analyzer.metrics_avg.to_dict() if analyzer.metrics_avg is not None else {},
            "results_summary": analyzer.results_summary,
        }

    except Exception as e:
        plt.show = original_show
        print(f"❌ 结果分析失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "plots_dir": None, "stats": {}}


# ====================== 主入口 ======================
def main():
    print_separator("SAR-VQA 主评估流程启动")

    entry = _MODEL_REGISTRY.get(MODEL_KEY)
    if not entry:
        print(f"❌ 无效的 MODEL_KEY: {MODEL_KEY}")
        return

    model_type = entry["model_type"]
    file_tag = entry["file_tag"]

    # ---- 校验流水线模式（full/eval/analyze）----
    resolved = resolve_pipeline_mode(model_type, file_tag)
    if resolved is None:
        sys.exit(1)
    mode, ref_file = resolved

    # ---- 生成统一时间戳和输出目录 ----
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg.path_config.RESULT_DIR, model_type, file_tag, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n📋 当前配置:")
    mode_desc = {"full": "预测 + 评估 + 分析", "eval": "评估 + 分析（跳过预测）",
                 "analyze": "仅分析（跳过预测与评估）"}[mode]
    print(f"   - 流水线模式: {mode}（{mode_desc}）")
    if ref_file:
        print(f"   - 引用历史结果: {ref_file}")
    print(f"   - 模型: {MODEL_KEY}")
    print(f"   - 模型类型: {model_type}")
    print(f"   - 文件标签: {file_tag}")
    print(f"   - 数据集: {DATASET_TAG}")
    print(f"   - 统一输出目录: {output_dir}")
    print(f"   - 图像路径: {IMAGE_BASE_PATH}")
    print(f"   - 最大行数: {MAX_PROCESS_ROWS} | 起始行: {START_ROW}")
    print(f"   - 最大并发: {MAX_WORKERS} | 批次阈值: {BATCH_SAVE_THRESHOLD}")
    print(f"   - 置信度阈值: {CONFIDENCE_THRESHOLD}")
    print(f"   - 目标指标: {TARGET_METRICS}")

    print_separator(char="-")

    if mode == "full":
        # ---- 三步流水线 ----
        print("\n" + "=" * 80)
        print("🚀 开始执行完整评估流程")
        print("=" * 80)

        pred_result = run_prediction(MODEL_KEY, output_dir)
        if not pred_result["success"]:
            print("❌ 模型预测失败，终止流程")
            update_excel_summary(timestamp, model_type, file_tag, output_dir,
                                 pred_result, {"success": False, "stats": {}},
                                 {"success": False, "stats": {}})
            return
        print("\n" + "=" * 80)

        bench_result = run_benchmark(file_tag, output_dir, pred_result.get("output_file"))
        if not bench_result["success"]:
            print("❌ 指标评估失败，终止流程")
            update_excel_summary(timestamp, model_type, file_tag, output_dir,
                                 pred_result, bench_result,
                                 {"success": False, "stats": {}})
            return
        print("\n" + "=" * 80)

        analysis_result = run_analysis(file_tag, output_dir, bench_result.get("output_file"))

        # ---- 更新 Excel 统计表 ----
        update_excel_summary(timestamp, model_type, file_tag, output_dir,
                             pred_result, bench_result, analysis_result)

        if analysis_result["success"]:
            print("\n" + "=" * 80)
            print("🎉 完整评估流程结束！")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("⚠️ 评估流程结束（结果分析失败）")
            print("=" * 80)


if __name__ == "__main__":
    main()
