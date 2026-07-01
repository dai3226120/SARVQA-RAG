"""
模型预测统一入口
整合 MainAgent 调用 + 并行数据处理流水线
"""

import os
import sys
import re
import time
import datetime
import threading
import concurrent.futures

import pandas as pd

from config import data_config, path_config, prompt_config
from agent.utils.thread_lock import lock
from agent.utils.print_utils import safe_print, format_elapsed_time
from agent.utils.text_utils import tokenize_text, clean_special_chars
from agent.utils.csv_utils import build_image_path, save_results_to_csv
from .metrics import retrieval_metrics, compute_information_metrics
from .benchmarker import Benchmarker


# ====================== MainAgent 客户端 ======================
class MainAgentClient:
    """MainAgent 客户端，支持 mainagent 和 mainagent_rscsv 两种变体"""

    # 支持的 Agent 模块映射
    AGENT_MODULES = {
        "mainagent": "agent.mainagent",
        "mainagent_internVL": "agent.mainagent_internVL",
        "mainagent_rscsv": "agent.mainagent_rscsv",
        "mainagent_internVL_rscsv": "agent.mainagent_internVL_rscsv",
    }

    def __init__(self, agent_module_name="mainagent"):
        """
        初始化 MainAgent 客户端

        参数:
            agent_module_name: Agent 模块名称
                可选: "mainagent" / "mainagent_internVL" / "mainagent_rscsv" / "mainagent_internVL_rscsv"
        """
        self.agent_module_name = agent_module_name
        self.agent = None
        self._initialized = False
        self.call_count = 0
        self.success_count = 0
        self.total_latency = 0.0
        self._rscsv_service_class = None
        self._tool_latency_tracker_class = None
        self._thread_local = threading.local()
        self._module_path = self.AGENT_MODULES.get(agent_module_name, f"agent.{agent_module_name}")

    def _init_agent(self):
        """初始化 Agent 实例（线程安全）"""
        if self._initialized:
            return

        init_messages = []
        exception = None

        with lock:
            if not self._initialized:
                init_messages.append(f"[INIT] 初始化 {self.agent_module_name} Agent 实例...")

                try:
                    import importlib

                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    parent_dir = os.path.dirname(current_dir)  # benchmark/

                    # 记录哪些路径原本在 sys.path 中，临时移除避免与 Agent 包冲突
                    was_current_in_path = current_dir in sys.path
                    was_parent_in_path = parent_dir in sys.path
                    cached_modules = {}

                    if was_current_in_path:
                        sys.path.remove(current_dir)
                    if was_parent_in_path:
                        sys.path.remove(parent_dir)

                    # 清除 benchmark/utils.py 的模块缓存，让 Agent 能导入自己的 utils/ 包
                    benchmark_utils_keys = [key for key in sys.modules.keys()
                                            if key.startswith('utils') or key.startswith('config')]
                    for mod_name in benchmark_utils_keys:
                        cached_modules[mod_name] = sys.modules.pop(mod_name, None)

                    if path_config.AGENT_ROOT_DIR not in sys.path:
                        sys.path.insert(0, path_config.AGENT_ROOT_DIR)

                    init_messages.append(f"   - Agent 根目录: {path_config.AGENT_ROOT_DIR}")
                    init_messages.append(f"   - 模块路径: {self._module_path}")

                    agent_module = importlib.import_module(self._module_path)
                    MainAgent = getattr(agent_module, "MainAgent")

                    init_messages.append("   - 正在创建 Agent 实例...")

                    # 重置隶属度统计
                    from rag.rag_rscsv_service import RscsvService
                    self._rscsv_service_class = RscsvService
                    if hasattr(RscsvService, 'reset_membership_stats'):
                        RscsvService.reset_membership_stats()
                    elif hasattr(RscsvService, '_membership_total_calls'):
                        RscsvService._membership_total_calls = 0
                        RscsvService._membership_hit_calls = 0

                    # 获取工具耗时追踪器类引用
                    from tools.middleware import ToolLatencyTracker
                    self._tool_latency_tracker_class = ToolLatencyTracker
                    ToolLatencyTracker.reset_global()

                    self.agent = MainAgent()
                    self._initialized = True
                    init_messages.append(f"[OK] {self.agent_module_name} Agent 初始化成功")

                except ImportError as e:
                    init_messages.append(f"[ERROR] 导入 Agent 模块失败: {e}")
                    exception = e
                finally:
                    if was_current_in_path and current_dir not in sys.path:
                        sys.path.insert(0, current_dir)
                    if was_parent_in_path and parent_dir not in sys.path:
                        sys.path.insert(0, parent_dir)
                    sys.modules.update(cached_modules)

        for msg in init_messages:
            safe_print(msg)

        if exception:
            raise exception

    def call(self, image_path, question, prompt_template=None):
        """
        调用 MainAgent 进行预测

        参数:
            image_path: 图像文件路径
            question: 问题文本
            prompt_template: 提示词模板（可选，默认使用 Agent 提示词）

        返回:
            str: 过滤后的纯答案
        """
        call_start_time = time.time()
        self.call_count += 1
        image_name = os.path.basename(image_path)

        if not os.path.exists(image_path):
            safe_print(f"[WARN] 图像文件不存在: {image_name}")
            return "Image file not found"

        try:
            self._init_agent()

            if self._tool_latency_tracker_class:
                self._tool_latency_tracker_class.reset_session()

            template = prompt_template or prompt_config.AGENT_PROMPT
            formatted_prompt = template.format(question=question)

            safe_print(f"[CALL] 调用 Agent [{self.call_count}] | 图像: {image_name}")

            with open(image_path, 'rb') as image_file:
                response_chunks = []
                for chunk in self.agent.execute_stream(query=formatted_prompt, image_file=image_file):
                    response_chunks.append(chunk)
                full_response = ''.join(response_chunks).strip()

            call_latency = time.time() - call_start_time
            self.total_latency += call_latency
            self.success_count += 1

            retrieval_latency = 0.0
            if self._tool_latency_tracker_class:
                retrieval_latency = self._tool_latency_tracker_class.get_session_retrieval_latency()
            self._thread_local.last_retrieval_latency = retrieval_latency

            safe_print(f"[OK] Agent 调用成功 [{self.call_count}] | 耗时: {call_latency:.2f}秒 | 检索耗时: {retrieval_latency:.4f}秒")

            ig_value, id_value = compute_information_metrics(question, formatted_prompt)
            retrieval_metrics.add_ig_id(ig_value, id_value)
            retrieval_metrics.update_from_response(full_response)

            filtered_answer = self._filter_tool_logs(full_response)
            return filtered_answer

        except Exception as e:
            call_latency = time.time() - call_start_time
            error_msg = f"Error: {str(e)}"

            retrieval_latency = 0.0
            if self._tool_latency_tracker_class:
                retrieval_latency = self._tool_latency_tracker_class.get_session_retrieval_latency()
            self._thread_local.last_retrieval_latency = retrieval_latency

            safe_print(f"[ERROR] Agent 调用失败 [{self.call_count}] | 图像: {image_name} | 耗时: {call_latency:.2f}秒 | 检索耗时: {retrieval_latency:.4f}秒 | 错误: {error_msg}")
            return error_msg

    def get_stats(self):
        """获取调用统计信息"""
        avg_latency = self.total_latency / self.call_count if self.call_count > 0 else 0.0
        success_rate = self.success_count / self.call_count * 100 if self.call_count > 0 else 0.0
        return {
            "call_count": self.call_count,
            "success_count": self.success_count,
            "failed_count": self.call_count - self.success_count,
            "success_rate": success_rate,
            "total_latency": self.total_latency,
            "avg_latency": avg_latency
        }

    def print_stats(self):
        """打印调用统计信息"""
        stats = self.get_stats()
        safe_print("")
        safe_print("=" * 60)
        safe_print(f"Agent 调用统计 - {self.agent_module_name}")
        safe_print("=" * 60)
        safe_print(f"   - 总调用次数: {stats['call_count']}")
        safe_print(f"   - 成功次数: {stats['success_count']}")
        safe_print(f"   - 失败次数: {stats['failed_count']}")
        safe_print(f"   - 成功率: {stats['success_rate']:.2f}%")
        safe_print(f"   - 总耗时: {self.total_latency:.2f}秒")
        safe_print(f"   - 平均耗时: {stats['avg_latency']:.2f}秒/次")

        latency_stats = self.get_retrieval_latency_stats()
        if latency_stats["call_count"] > 0:
            safe_print("-" * 60)
            safe_print(f"检索工具统计:")
            safe_print(f"   - 调用次数: {latency_stats['call_count']}")
            safe_print(f"   - 总耗时: {latency_stats['total_latency']:.4f}秒")
            safe_print(f"   - 平均耗时: {latency_stats['avg_latency']:.4f}秒/次")

        safe_print("=" * 60)

    def _filter_tool_logs(self, full_response):
        """过滤工具调用日志，只保留纯回答内容"""
        FILTER_PREFIXES = [
            "rag_rscsv", "rag_summarize",
            "【匹配隶属度缓存】", "综合隶属度得分=",
            "q: ", "a: "
        ]

        filtered_lines = []
        for line in full_response.split('\n'):
            line = line.strip()
            if any(prefix in line.lower() for prefix in FILTER_PREFIXES):
                continue
            if not line:
                continue
            filtered_lines.append(line)

        pure_answer = ' '.join(filtered_lines).strip()

        if not pure_answer:
            parts = re.split(r'【|】|\(|\)|：|:', full_response)
            pure_answer = parts[-1].strip() if parts else full_response

        pure_answer = pure_answer.replace("--- [Tool Output:", "").replace("] ---", "")
        pure_answer = pure_answer.replace("---------------------------", "")
        pure_answer = ' '.join(pure_answer.split()[:200])

        return pure_answer

    def get_tool_hit_stats(self):
        """获取工具命中率统计"""
        if self.agent:
            return self.agent.get_tool_hit_stats()
        return {}

    def get_rag_rscsv_membership_hit_rate(self):
        """获取 rag_rscsv 的真正隶属度命中率"""
        if self._rscsv_service_class:
            try:
                return self._rscsv_service_class.get_membership_hit_rate_static()
            except Exception as e:
                safe_print(f"[WARN] 获取隶属度命中率失败: {e}")
                return None
        return None

    def get_last_retrieval_latency(self):
        """获取当前线程最后一次调用的检索耗时"""
        return getattr(self._thread_local, 'last_retrieval_latency', 0.0)

    def get_retrieval_latency_stats(self):
        """获取全局检索耗时统计"""
        if self._tool_latency_tracker_class:
            return {
                "total_latency": self._tool_latency_tracker_class.get_global_total_latency(),
                "call_count": self._tool_latency_tracker_class.get_global_call_count(),
                "avg_latency": self._tool_latency_tracker_class.get_global_avg_latency()
            }
        return {"total_latency": 0.0, "call_count": 0, "avg_latency": 0.0}

    def reset_retrieval_latency_stats(self):
        """重置全局检索耗时统计"""
        if self._tool_latency_tracker_class:
            self._tool_latency_tracker_class.reset_global()


# ====================== 全局 Agent 实例 ======================
doubao_agent_client = MainAgentClient("mainagent")
internvl_agent_client = MainAgentClient("mainagent_internVL")
doubao_agent_rscsv_client = MainAgentClient("mainagent_rscsv")
internvl_agent_rscsv_client = MainAgentClient("mainagent_internVL_rscsv")


# ====================== 便捷函数 ======================
def call_doubao_agent(image_path, question, prompt_template=None):
    """便捷函数：调用豆包 Agent"""
    return doubao_agent_client.call(image_path, question, prompt_template)


def call_internvl_agent(image_path, question, prompt_template=None):
    """便捷函数：调用 InternVL Agent"""
    return internvl_agent_client.call(image_path, question, prompt_template)


# ====================== 并行数据处理流水线 ======================
def process_vqa_data(
    process_func,
    csv_input_path,
    result_dir,
    base_filename,
    image_base_path,
    max_rows=None,
    max_workers=None,
    start_row=None,
    required_columns=None
):
    """
    通用的 SAR-VQA 数据并行处理函数

    参数:
        process_func: 处理单行数据的函数，签名为 (index, row, image_base_path) -> dict
        csv_input_path: 输入 CSV 路径
        result_dir: 结果目录
        base_filename: 结果文件前缀
        image_base_path: 图像基础路径
        max_rows: 最大处理行数
        max_workers: 最大并发数
        start_row: 起始行
        required_columns: 必需列名列表
    """
    safe_print("=" * 80)
    safe_print("[START] 开始处理 SAR-VQA 数据")
    safe_print("=" * 80)

    if max_rows is None:
        max_rows = data_config.MAX_PROCESS_ROWS
    if max_workers is None:
        max_workers = data_config.MAX_WORKERS
    if start_row is None:
        start_row = data_config.START_ROW
    if required_columns is None:
        required_columns = data_config.REQUIRED_COLUMNS

    safe_print("[CONFIG] 处理配置:")
    safe_print(f"  - 输入文件: {csv_input_path}")
    safe_print(f"  - 结果目录: {result_dir}")
    safe_print(f"  - 图像路径: {image_base_path}")
    safe_print(f"  - 最大行数: {max_rows}")
    safe_print(f"  - 起始行: {start_row}")
    safe_print(f"  - 最大并发: {max_workers}")
    safe_print(f"  - 批次阈值: {data_config.BATCH_SAVE_THRESHOLD}")
    safe_print(f"  - 必需列: {required_columns}")
    safe_print("-" * 80)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv_path = f"{result_dir}/{base_filename}_{timestamp}.csv"
    latest_csv_path = f"{result_dir}/{base_filename}_latest.csv"

    os.makedirs(result_dir, exist_ok=True)

    if not os.path.exists(csv_input_path):
        safe_print(f"[ERROR] CSV 文件不存在: {csv_input_path}")
        return None

    try:
        safe_print("[INFO] 正在读取 CSV 文件...")
        if start_row == 0:
            df = pd.read_csv(csv_input_path, nrows=max_rows, encoding=data_config.CSV_READ_ENCODING)
        else:
            df = pd.read_csv(
                csv_input_path,
                skiprows=range(1, start_row),
                nrows=max_rows,
                encoding=data_config.CSV_READ_ENCODING
            )
        df.columns = df.columns.str.strip()

        safe_print("[OK] 成功读取 CSV 文件")
        safe_print(f"   - 总行数: {len(df)}")
        safe_print(f"   - 列名: {df.columns.tolist()}")

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            safe_print(f"[ERROR] 缺少必要列: {missing_columns}")
            return None

    except Exception as e:
        safe_print(f"[ERROR] 读取 CSV 文件失败: {e}")
        return None

    results = []
    total_tasks = len(df)
    start_time = time.time()
    failed_count = 0

    safe_print(f"\n[PROCESS] 开始并行处理 ({total_tasks} 条任务)")
    safe_print(f"   线程数: {max_workers}")
    safe_print("-" * 80)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_row = {
            executor.submit(process_func, idx + start_row, row, image_base_path): (idx + start_row, row)
            for idx, row in df.iterrows()
        }

        for completed_count, future in enumerate(concurrent.futures.as_completed(future_to_row), start=1):
            idx, row = future_to_row[future]
            try:
                result_item = future.result()
                if result_item is not None:
                    results.append(result_item)

                if len(results) % data_config.BATCH_SAVE_THRESHOLD == 0 or completed_count == total_tasks:
                    elapsed = time.time() - start_time
                    speed = completed_count / elapsed if elapsed > 0 else 0.0
                    progress = completed_count / total_tasks * 100 if total_tasks else 100.0
                    remaining = (total_tasks - completed_count) / speed if speed > 0 else 0

                    safe_print(f"\n[PROGRESS] 进度更新 [{completed_count}/{total_tasks}]")
                    safe_print(f"   进度: {progress:.1f}%")
                    safe_print(f"   速度: {speed:.2f} 条/秒")
                    safe_print(f"   已用时: {format_elapsed_time(elapsed)}")
                    safe_print(f"   预计剩余: {format_elapsed_time(remaining)}")
                    safe_print(f"   已完成: {len(results)}条有效记录 | 失败: {failed_count}条")

                    safe_print(f"\n[SAVE] 保存批次结果...")
                    save_results_to_csv(results, output_csv_path, latest_csv_path)
                    safe_print(f"   - {output_csv_path}")
                    safe_print(f"   - {latest_csv_path}")
                    safe_print("-" * 80)

            except Exception as e:
                failed_count += 1
                safe_print(f"[ERROR] 第{idx + 1}行出错: {str(e)}")

    try:
        result_df = save_results_to_csv(results, output_csv_path, latest_csv_path)

        elapsed = time.time() - start_time
        avg_per_input = elapsed / total_tasks if total_tasks else 0.0
        avg_per_output = elapsed / len(results) if results else 0.0
        success_rate = len(results) / total_tasks * 100 if total_tasks else 0

        safe_print("\n" + "=" * 80)
        safe_print("[DONE] 处理完成!")
        safe_print("=" * 80)
        safe_print("[STATS] 处理统计:")
        safe_print(f"   - 总任务数: {total_tasks}")
        safe_print(f"   - 成功: {len(results)}条")
        safe_print(f"   - 失败: {failed_count}条")
        safe_print(f"   - 成功率: {success_rate:.2f}%")
        safe_print("\n[TIME] 时间统计:")
        safe_print(f"   - 总时长: {format_elapsed_time(elapsed)}")
        safe_print(f"   - 平均耗时: {avg_per_input:.3f}秒/条")
        safe_print("\n[OUTPUT] 输出文件:")
        safe_print(f"   - {output_csv_path}")
        safe_print(f"   - {latest_csv_path}")
        safe_print("=" * 80)

        return result_df

    except Exception as e:
        safe_print(f"[ERROR] 保存失败: {e}")
        return None


def create_process_row_func(api_call_func, include_metrics=False):
    """
    创建处理单行数据的函数

    参数:
        api_call_func: API 调用函数，签名为 (image_path, question) -> str
        include_metrics: 是否包含指标计算
    """
    def process_single_row(index, row, image_base_path):
        try:
            id_text = clean_special_chars(row['id'])
            image_path = clean_special_chars(row['image'])
            question_text = clean_special_chars(row['question'])
            answer_text = clean_special_chars(row['answer'])

            image_path_front = build_image_path(image_path, image_base_path)

            predicted_answer = api_call_func(image_path_front, question_text)
            predicted_answer = clean_special_chars(predicted_answer)
            predicted_answer = predicted_answer.replace('\n', ' ').replace('\r', ' ').strip()

            result = {
                "id": id_text,
                "image": image_path,
                "question": question_text.lower(),
                "answer": answer_text.lower(),
                "predicted": predicted_answer.lower()
            }

            if include_metrics:
                formatted_prompt = prompt_config.DEFAULT_PROMPT.format(question=question_text)
                ig_value, id_value = compute_information_metrics(question_text, formatted_prompt)
                result["IG"] = ig_value
                result["ID"] = id_value
                retrieval_metrics.add_ig_id(ig_value, id_value)

            return result

        except KeyError as e:
            safe_print(f"[ERROR] 第{index + 1}行数据访问错误: 列{e}不存在")
            return None
        except Exception as e:
            safe_print(f"[ERROR] 第{index + 1}行处理出错: {str(e)}")
            return None

    return process_single_row
