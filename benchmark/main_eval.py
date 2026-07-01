"""
SAR-VQA 主评估脚本 - 统一调用入口
通过 config / core / models 三个包实现预测→评估→分析流水线
"""

import os
import sys
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
    MainAgentClient,
    doubao_agent_client,
    internvl_agent_client,
    doubao_agent_rscsv_client,
    internvl_agent_rscsv_client,
    process_vqa_data,
    create_process_row_func,
    Benchmarker,
    ResultAnalyzer,
)
from models import doubao_client, internvl_client


# ====================== 模型映射配置 ======================
# 每个 MODEL_KEY 对应：客户端实例 / 调用函数 / 文件标签
MODEL_KEY = "agent-text-doubao-seed_rscsv"
# MODEL_KEY = "agent-text-doubao-seed"

_MODEL_REGISTRY = {
    "doubao-seed": {
        "file_tag": cfg.get_file_tag(cfg.ModelType.DOUBAO),
        "client": doubao_client,
        "api_call": doubao_client.call,
    },
    "internVL": {
        "file_tag": cfg.get_file_tag(cfg.ModelType.INTERNVL),
        "client": internvl_client,
        "api_call": internvl_client.call,
    },
    "agent-text-doubao-seed": {
        "file_tag": cfg.get_file_tag(cfg.ModelType.AGENT_DOUBAO),
        "client": doubao_agent_client,
        "api_call": doubao_agent_client.call,
    },
    "agent-text-doubao-seed_rscsv": {
        "file_tag": "agent-text-doubao-seed-2-0-mini_rscsv",
        "client": doubao_agent_rscsv_client,
        "api_call": doubao_agent_rscsv_client.call,
    },
    "agent-text-internVL": {
        "file_tag": cfg.get_file_tag(cfg.ModelType.AGENT_INTERNVL),
        "client": internvl_agent_client,
        "api_call": internvl_agent_client.call,
    },
}

DATASET_TAG = "val"
IMAGE_BASE_PATH = cfg.path_config.IMAGE_BASE_PATH
CONFIDENCE_THRESHOLD = cfg.analysis_config.CONFIDENCE_THRESHOLD
TARGET_METRICS = cfg.analysis_config.TARGET_METRICS


# ====================== 辅助函数 ======================
def print_separator(title=None, char="="):
    if title:
        print(f"{char} {title} {char}")
        print(char * 80)
    else:
        print(char * 80)


# ====================== 步骤 1：模型预测 ======================
def run_prediction(model_key: str) -> bool:
    """使用 core.predictor 执行模型预测"""
    entry = _MODEL_REGISTRY.get(model_key)
    if not entry:
        print(f"❌ 未知模型: {model_key}")
        return False

    file_tag = entry["file_tag"]
    api_call = entry["api_call"]
    client = entry["client"]

    print_separator(f"步骤 1/3: 模型预测 ({model_key})")

    csv_input_path = cfg.get_csv_input_path(DATASET_TAG)
    result_dir = cfg.get_result_dir(file_tag)
    base_filename = cfg.get_base_filename(file_tag, DATASET_TAG)

    print(f"\n📋 预测配置:")
    print(f"   - 模型: {model_key}")
    print(f"   - 文件标签: {file_tag}")
    print(f"   - 输入 CSV: {csv_input_path}")
    print(f"   - 结果目录: {result_dir}")
    print(f"   - 图像路径: {IMAGE_BASE_PATH}")

    # 构建处理函数
    process_func = create_process_row_func(api_call, include_metrics=True)

    print_separator(char="-")

    try:
        result_df = process_vqa_data(
            process_func=process_func,
            csv_input_path=csv_input_path,
            result_dir=result_dir,
            base_filename=base_filename,
            image_base_path=IMAGE_BASE_PATH,
        )

        if result_df is not None:
            # 打印客户端统计
            if hasattr(client, 'print_stats'):
                client.print_stats()
            print(f"✅ 模型预测完成")
            return True
        else:
            print(f"❌ 模型预测返回空结果")
            return False

    except Exception as e:
        print(f"❌ 模型预测失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ====================== 步骤 2：指标评估 ======================
def run_benchmark(file_tag: str) -> bool:
    """使用 core.benchmarker.Benchmarker 执行指标评估"""
    print_separator(f"步骤 2/3: 指标评估 (FILE_TAG={file_tag})")

    input_csv = f"./benchmark/result/{file_tag}/{file_tag}_{DATASET_TAG}_predicted_question_latest.csv"
    result_dir = cfg.path_config.BENCHMARK_RESULT_DIR
    base_filename = f"{file_tag}_{DATASET_TAG}_benchmark"

    print(f"\n📋 评估配置:")
    print(f"   - 输入: {input_csv}")
    print(f"   - 输出目录: {result_dir}")
    print(f"   - 基础文件名: {base_filename}")

    print_separator(char="-")

    try:
        benchmarker = Benchmarker(
            input_csv_path=input_csv,
            result_dir=result_dir,
            base_filename=base_filename,
        )

        if not benchmarker.check_input_csv_structure():
            print("❌ 输入 CSV 结构检查失败")
            return False

        output_path = benchmarker.run()
        if output_path:
            print(f"✅ 指标评估完成 → {output_path}")
            return True
        else:
            print(f"❌ 指标评估失败")
            return False

    except Exception as e:
        print(f"❌ 指标评估失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ====================== 步骤 3：结果分析 ======================
def run_analysis(file_tag: str) -> bool:
    """使用 core.analyzer.ResultAnalyzer 执行结果分析"""
    print_separator(f"步骤 3/3: 结果分析 (FILE_TAG={file_tag})")

    plt_save_dir = os.path.join(cfg.path_config.BENCHMARK_RESULT_DIR, f"{file_tag}_plots")
    os.makedirs(plt_save_dir, exist_ok=True)
    print(f"📁 图表保存目录: {plt_save_dir}")

    base_filename = f"{file_tag}_{DATASET_TAG}_benchmark"
    csv_path = f"{cfg.path_config.BENCHMARK_RESULT_DIR}/{base_filename}_latest.csv"

    print(f"\n📋 分析配置:")
    print(f"   - 输入: {csv_path}")
    print(f"   - 置信度阈值: {CONFIDENCE_THRESHOLD}")
    print(f"   - 目标指标: {TARGET_METRICS}")

    print_separator(char="-")

    # Monkey-patch plt.show → 保存图片
    original_show = plt.show
    plot_counter = [0]

    def save_and_close(*args, **kwargs):
        filename = f"plot_{plot_counter[0]}_{file_tag}.png"
        filepath = os.path.join(plt_save_dir, filename)
        plt.savefig(filepath, dpi=100, bbox_inches='tight', format='png')
        print(f"📊 图表已保存: {filename}")
        plt.close()
        plot_counter[0] += 1

    plt.show = save_and_close

    try:
        analyzer = ResultAnalyzer(
            csv_path=csv_path,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            target_metrics=TARGET_METRICS,
        )
        analyzer.analyze(plot=True, save_dir=plt_save_dir)

        plt.show = original_show
        print(f"\n✅ 结果分析完成")
        print(f"📁 图表已保存至: {plt_save_dir}")
        return True

    except Exception as e:
        plt.show = original_show
        print(f"❌ 结果分析失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ====================== 主入口 ======================
def main():
    print_separator("SAR-VQA 主评估流程启动")

    entry = _MODEL_REGISTRY.get(MODEL_KEY)
    if not entry:
        print(f"❌ 无效的 MODEL_KEY: {MODEL_KEY}")
        return

    file_tag = entry["file_tag"]

    print(f"\n📋 当前配置:")
    print(f"   - 模型: {MODEL_KEY}")
    print(f"   - 文件标签: {file_tag}")
    print(f"   - 数据集: {DATASET_TAG}")
    print(f"   - 图像路径: {IMAGE_BASE_PATH}")
    print(f"   - 置信度阈值: {CONFIDENCE_THRESHOLD}")
    print(f"   - 目标指标: {TARGET_METRICS}")

    print_separator(char="-")

    # ---- 三步流水线 ----
    print("\n" + "=" * 80)
    print("🚀 开始执行完整评估流程")
    print("=" * 80)

    if not run_prediction(MODEL_KEY):
        print("❌ 模型预测失败，终止流程")
        return
    print("\n" + "=" * 80)

    if not run_benchmark(file_tag):
        print("❌ 指标评估失败，终止流程")
        return
    print("\n" + "=" * 80)

    if not run_analysis(file_tag):
        print("❌ 结果分析失败")
    else:
        print("\n" + "=" * 80)
        print("🎉 完整评估流程结束！")
        print("=" * 80)


if __name__ == "__main__":
    main()
