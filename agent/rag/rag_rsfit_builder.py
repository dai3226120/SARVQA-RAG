import os
import base64
import re
import pandas as pd
import numpy as np
import math
import hashlib
import requests
import threading
import time
from datetime import datetime
from collections import defaultdict
from sklearn.cluster import DBSCAN
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 导入LangChain的Chroma向量数据库集成
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
# 从工厂模块导入嵌入模型和多模态模型
from model.factory import huggingface_embed_model, doubao_seed_20_mini_model
from utils.config_handler import chroma_conf
from rag.rag_rscsv_builder import RscsvBuilder
from utils.file_handler import get_file_md5_hex
from utils.path_tool import get_abs_path
from utils.data_process import image_to_base64, normalize_text
from utils.logger_handler import logger


# ================================================================
# 大模型判断配置（从benchmark copy.py引入）
# ================================================================
LLM_API_KEY = "e9f2dacd-d0a2-4c9a-ba2a-805ee0b40dcd"
LLM_API_ENDPOINT = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
LLM_MODEL_NAME = "doubao-1-5-lite-32k-250115"
LLM_API_TIMEOUT = 30
LLM_API_TEMPERATURE = 0.7
LLM_SYSTEM_PROMPT = "你是一个语义匹配判断助手，仅返回1或0，不解释原因"
LLM_PROMPT_TEMPLATE = """
Question: {question}
Ground Truth Answer: {answer}
Predicted Answer: {predicted}
Does the predicted answer match the ground truth? Answer 1 for match and 0 for not match. 
Use semantic meaning not exact match. Synonyms are also treated as a match, e.g., football and soccer, 
playground and ground track field, building and rooftop, pond and swimming pool. Do not explain the reason.
""".strip()
LLM_LOCK = threading.Lock()
LLM_MAX_WORKERS = 50  # 并发线程数


# ================================================================
# 运行模式配置（通过变量控制，而非命令行参数）
# ================================================================
# 可选值：
#   - "incremental": 增量更新模式（默认）- 加载已有日志，MD5校验后决定是否重载向量库
#   - "force_reload": 强制全量重载模式 - 清空CSV和向量库，重新开始
#   - "rejudge": 重判模式 - 清除向量库，重新对rag_feedback_logs.csv进行评估并入库
RUN_MODE = "incremental"

# ================================================================
# 可配置参数
# ================================================================
# RAG 检索配置
RETRIEVAL_K = chroma_conf["k"]                          # 检索返回的信息切片数量
MEMBERSHIP_K = chroma_conf["membership_k"]              # 隶属度计算时检索的日志条目数
SLICE_K = chroma_conf["slice_k"]              # 基础切片检索时检索的日志条目数
TOP_P = chroma_conf["top_p"]                          # 取 top_p 条结果保留数

# RETRIEVAL_K = 50000                          # 测试用
# MEMBERSHIP_K = 50000                          # 测试用
MEMBERSHIP_W1 = chroma_conf["w1"]                     # 隶属度权重1：相似度权重
MEMBERSHIP_W2 = chroma_conf["w2"]                       # 隶属度权重2：正确性分数权重

# 相似度评估配置
BLEU_WEIGHT = chroma_conf["bleu_weight"]                        # BLEU分数权重
OVERLAP_WEIGHT = chroma_conf["overlap_weight"]                     # 词汇重叠权重

# 测试配置
TEST_NROWS = 50000                           # 读取的测试数据行数
BASE_IMAGE_DIR = r"C:\dataset\SAR-TEXT"   # 测试图片基础目录

# 日志配置
LOG_NAME = "rag_feedback_logs.csv"        # RAG测试日志文件名
ERROR_LOG_NAME = "rag_error_logs.csv"     # LLM判断错误的日志文件名
COLLECTION_NAME = "rag_test_logs"         # 向量数据库集合名

# ================================================================
# 配置类（如需更复杂的配置管理可扩展）
# ================================================================
class Config:
    """系统运行配置"""

    # RAG 检索
    retrieval_k = RETRIEVAL_K
    membership_k = MEMBERSHIP_K
    slice_k = SLICE_K
    top_p = TOP_P
    membership_w1 = MEMBERSHIP_W1
    membership_w2 = MEMBERSHIP_W2

    # 相似度评估
    bleu_weight = BLEU_WEIGHT
    overlap_weight = OVERLAP_WEIGHT

    # 测试
    test_nrows = TEST_NROWS
    base_image_dir = BASE_IMAGE_DIR

    # 日志
    log_name = LOG_NAME
    error_log_name = ERROR_LOG_NAME
    collection_name = COLLECTION_NAME

    # VLM Agent 配置
    vlm_agent_type = chroma_conf.get("vlm_agent_type", "doubao")  # doubao 或 internvl


class SARSemanticCacheSystem:
    """
    SAR RAG 测试与隶属度计算系统

    核心流程：
    1. evaluate_and_log: 对测试数据进行RAG检索+多模态模型预测，评估正确性并记录日志
    2. calculate_membership_degree: 新问题与日志做隶属度计算，获得隶属度得分和推荐切片
    """

    def __init__(self, force_full_reload: bool = False, clear_vector_db_only: bool = False):
        """
        初始化系统
        
        Args:
            force_full_reload: 强制全量重载模式 - 清空CSV和向量库，重新开始
            clear_vector_db_only: 只清除向量库模式 - 保留CSV，清除向量库（用于rejudge模式）
        """
        # ========================================
        # 1. 初始化嵌入模型（用于向量化和相似度计算）
        # ========================================
        self.embeddings = huggingface_embed_model

        # ========================================
        # 2. 初始化日志向量库（存储RAG测试日志用于隶属度计算）
        # ========================================
        self.persist_directory = get_abs_path(chroma_conf["persist_diretory"])
        self.logs_collection = Chroma(
            collection_name=Config.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space": "cosine"}
        )

        # ========================================
        # 3. 初始化多模态模型（用于图像+文本理解和预测）
        # ========================================
        self.multimodal_llm = doubao_seed_20_mini_model

        # ========================================
        # 3.5 VLM Agent配置（懒加载，首次使用时初始化，避免循环导入）
        # ========================================
        self.vlm_agent = None
        self._vlm_agent_initialized = False
        self.vlm_agent_type = Config.vlm_agent_type
        logger.info(f"[VLM Agent] 配置类型: {self.vlm_agent_type}（懒加载）")

        # ========================================
        # 4. 文件路径配置
        # ========================================
        self.data_dir = get_abs_path(chroma_conf["data_path"])  # 数据存储目录
        self.log_name = Config.log_name
        self.log_path = os.path.join(self.data_dir, self.log_name)
        self.error_log_path = os.path.join(self.data_dir, Config.error_log_name)
        self.md5_path = get_abs_path(chroma_conf.get("md5_log_store", "md5_log.text"))
        os.makedirs(self.data_dir, exist_ok=True)  # 确保目录存在

        # ========================================
        # 5. 加载已有日志（根据模式选择不同策略）
        # ========================================
        if force_full_reload:
            # 模式1: 强制全量重载 - 清空CSV + 清空向量库
            logger.warning("[强制全量更新] force_full_reload=True，清空所有日志并重建...")

            # 清空反馈日志CSV
            self.log_df = pd.DataFrame(
                columns=['id', 'question', 'retrieved_slices', 'correctness_score', 'predicted_text', 'timestamp']
            )
            self.log_df.to_csv(self.log_path, index=False, encoding='utf-8')
            logger.info("[强制全量更新] 已清空反馈日志文件: %s", self.log_path)

            # 清空错误日志CSV
            error_df = pd.DataFrame(
                columns=['id', 'question', 'retrieved_slices', 'correctness_score', 'predicted_text',
                         'ground_truth', 'llm_judge', 'timestamp']
            )
            error_df.to_csv(self.error_log_path, index=False, encoding='utf-8')
            logger.info("[强制全量更新] 已清空错误日志文件: %s", self.error_log_path)

            # 清空并重建向量库（此时 log_df 为空，只会清空向量库）
            self._reload_logs_to_vector_db()
            logger.info("[强制全量更新] 向量库已清空并重建完成")

        elif clear_vector_db_only:
            # 模式3: 只清除向量库 - 保留CSV，加载CSV数据，但不清空向量库（后续手动调用 _reload）
            logger.warning("[仅清空向量库] clear_vector_db_only=True，保留CSV，清除向量库...")
            
            # 加载已有CSV数据
            if os.path.exists(self.log_path):
                self.log_df = pd.read_csv(self.log_path)
                # 确保所需列存在
                for col in ['id', 'question', 'retrieved_slices', 'correctness_score', 'predicted_text', 'timestamp']:
                    if col not in self.log_df.columns:
                        self.log_df[col] = ""
                
                # 去重日志数据
                self._deduplicate_log_df()
                logger.info(f"[仅清空向量库] 已加载 {len(self.log_df)} 条CSV记录")
            else:
                # CSV不存在，初始化空表
                self.log_df = pd.DataFrame(
                    columns=['id', 'question', 'retrieved_slices', 'correctness_score', 'predicted_text', 'timestamp']
                )
                logger.warning("[仅清空向量库] CSV文件不存在，初始化空表")
            
            # 清空向量库目录（后续在 run_real_test 中手动调用 _reload_logs_to_vector_db 入库）
            import shutil
            import gc
            
            if os.path.exists(self.persist_directory):
                try:
                    self.logs_collection._client.delete_collection(Config.collection_name)
                    logger.info("[仅清空向量库] 已通过 Chroma API 删除集合")
                except Exception as e:
                    logger.warning(f"[仅清空向量库] 通过 Chroma API 删除集合失败: {e}，尝试文件系统方式...")
                    del self.logs_collection
                    gc.collect()
                    shutil.rmtree(self.persist_directory)
                    logger.info("[仅清空向量库] 已通过文件系统删除向量库目录")
            
            os.makedirs(self.persist_directory, exist_ok=True)
            
            self.logs_collection = Chroma(
                collection_name=Config.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
                collection_metadata={"hnsw:space": "cosine"}
            )
            logger.info("[仅清空向量库] 向量库集合已重建（空）")

        elif os.path.exists(self.log_path):
            # 模式1: 增量更新（默认）- 加载已有日志 + MD5校验
            self.log_df = pd.read_csv(self.log_path)
            # 确保所需列存在
            for col in ['id', 'question', 'retrieved_slices', 'correctness_score', 'predicted_text', 'timestamp']:
                if col not in self.log_df.columns:
                    self.log_df[col] = ""
            
            # 关键修复1：去重日志数据，保留最新的记录
            self._deduplicate_log_df()
            
            # 执行 MD5 校验（决定是否需要重载向量库）
            self._check_and_reload_vector_db()
        else:
            # 初始化空日志表
            self.log_df = pd.DataFrame(
                columns=['id', 'question', 'retrieved_slices', 'correctness_score', 'predicted_text', 'timestamp']
            )

    def _init_vlm_agent(self):
        """根据配置初始化VLM Agent"""
        try:
            import importlib
            import sys

            # 将agent目录添加到Python路径，以便导入mainagent模块
            agent_dir = get_abs_path("agent")
            if agent_dir not in sys.path:
                sys.path.insert(0, agent_dir)

            if self.vlm_agent_type.lower() == "internvl":
                module = importlib.import_module("mainagent_internVL_rscsv")
                MainAgent = getattr(module, "MainAgent")
                self.vlm_agent = MainAgent()
                logger.info("[VLM Agent] 已初始化 mainagent_internVL_rscsv")
            else:
                module = importlib.import_module("mainagent_rscsv")
                MainAgent = getattr(module, "MainAgent")
                self.vlm_agent = MainAgent()
                logger.info("[VLM Agent] 已初始化 mainagent_rscsv")
        except Exception as e:
            logger.error(f"[VLM Agent] 初始化失败: {e}", exc_info=True)
            self.vlm_agent = None

    def call_vlm_agent(self, image_path, question, ground_truth=None):
        """
        调用VLM Agent获取预测结果

        Args:
            image_path: 图片路径
            question: 问题文本
            ground_truth: 参考答案（用于计算相似度得分）

        Returns:
            tuple: (score, response_text, slice_ids)
                - score: 正确性得分（BLEU+重叠）
                - response_text: 预测文本
                - slice_ids: agent实际使用的切片ID列表
        """
        # ==========================================
        # 阶段1：直接调用 RscsvServiceRscsv 获取切片
        # 确保与 rag_rscsv_service_rscsv.py 的切片获取逻辑完全一致
        # ==========================================
        from rag.rag_rscsv_service_rscsv import RscsvServiceRscsv
        rscsv_service = RscsvServiceRscsv()
        rscsv_result = rscsv_service.retrieve(question)
        
        # 从返回结果中提取切片ID（与 RscsvServiceRscsv.hybrid_retrieve 的格式一致）
        import re
        slice_ids_match = re.search(r'<!-- SLICE_IDS: (.*?) -->', rscsv_result)
        slice_ids = []
        if slice_ids_match:
            ids_str = slice_ids_match.group(1).strip()
            if ids_str:
                slice_ids = ids_str.split(',')
        logger.info(f"[VLM Agent] 从 RscsvServiceRscsv 获取的切片ID: {slice_ids}")

        if not image_path or not os.path.exists(image_path):
            logger.warning(f"[VLM Agent] 图片不存在: {image_path}")
            return 0.0, "", slice_ids

        try:
            # ==========================================
            # 阶段2：构建包含切片上下文的完整提示词
            # 将切片内容作为上下文传递给多模态模型
            # ==========================================
            enhanced_prompt = f"{rscsv_result}\n\n请结合以上参考资料和图片内容回答问题：{question}"

            # ==========================================
            # 阶段3：调用多模态模型进行预测
            # ==========================================
            score, response_text = self.get_vlm_prediction(image_path, enhanced_prompt, ground_truth)

        except Exception as e:
            logger.error(f"[VLM Agent] 调用失败: {e}", exc_info=True)
            score, response_text = self.get_vlm_prediction(image_path, question, ground_truth)

        return score, response_text, slice_ids

    def _deduplicate_log_df(self):
        """对日志数据进行去重，保留最新的记录（按timestamp排序）"""
        if self.log_df.empty:
            return
        
        original_length = len(self.log_df)
        
        # 转换timestamp列为datetime类型（处理格式异常）
        self.log_df['timestamp'] = pd.to_datetime(self.log_df['timestamp'], errors='coerce')
        
        # 按id分组，保留最新的记录
        self.log_df = self.log_df.sort_values('timestamp', ascending=False)
        self.log_df = self.log_df.drop_duplicates(subset=['id'], keep='first')
        self.log_df = self.log_df.reset_index(drop=True)
        
        # 去重后始终保存文件（确保内存中的数据持久化）
        self.log_df.to_csv(self.log_path, index=False, encoding='utf-8')
        
        if len(self.log_df) < original_length:
            logger.info(f"日志数据去重完成，移除 {original_length - len(self.log_df)} 条重复记录，当前有效记录数：{len(self.log_df)}")
        else:
            logger.info(f"日志数据去重完成，当前有效记录数：{len(self.log_df)}（无重复数据）")

    def call_llm_judge_api(self, question: str, answer: str, predicted: str) -> str:
        """
        调用大模型API判断预测答案是否与标准答案语义匹配

        Args:
            question: 问题文本
            answer: 标准答案
            predicted: 预测答案

        Returns:
            str: '1'（匹配）或 '0'（不匹配）
        """
        formatted_prompt = LLM_PROMPT_TEMPLATE.format(
            question=question,
            answer=answer,
            predicted=predicted
        )

        data = {
            "model": LLM_MODEL_NAME,
            "temperature": float(LLM_API_TEMPERATURE),
            "messages": [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": formatted_prompt}
            ],
            "thinking": {"type": "disabled"}
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API_KEY}"
        }

        try:
            response = requests.post(
                LLM_API_ENDPOINT,
                json=data,
                headers=headers,
                timeout=LLM_API_TIMEOUT
            )

            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    message_content = result['choices'][0].get('message', {}).get('content', '').strip()
                    if message_content not in ["0", "1"]:
                        message_content = "1" if "1" in message_content else "0" if "0" in message_content else "0"
                    return message_content
                else:
                    with LLM_LOCK:
                        logger.warning(f"LLM API返回无有效内容：{result}")
                    return "0"
            else:
                with LLM_LOCK:
                    logger.warning(f"LLM API调用失败，状态码：{response.status_code}, 错误：{response.text}")
                return "0"

        except requests.exceptions.RequestException as e:
            with LLM_LOCK:
                logger.warning(f"LLM API请求异常：{e}")
            return "0"
        except Exception as e:
            with LLM_LOCK:
                logger.warning(f"处理LLM API响应时发生错误：{e}")
            return "0"

    def llm_judge_and_filter(self, test_df: pd.DataFrame, new_logs: list, max_workers: int = LLM_MAX_WORKERS) -> tuple:
        """
        对新产生的日志进行大模型判断，过滤出判断错误的内容（并行版本）

        Args:
            test_df: 原始测试数据DataFrame（包含answer列）
            new_logs: 新产生的日志列表
            max_workers: 最大并发线程数，默认使用 LLM_MAX_WORKERS 配置

        Returns:
            tuple: (correct_logs, error_logs)
                - correct_logs: LLM判断正确的日志列表
                - error_logs: LLM判断错误的日志列表
        """
        import concurrent.futures

        if not new_logs:
            return [], []

        # 构建ID到标准答案的映射
        answer_map = {}
        for _, row in test_df.iterrows():
            answer_map[str(row['id'])] = str(row['answer']) if pd.notna(row['answer']) else ""

        correct_logs = []
        error_logs = []
        total = len(new_logs)
        start_time = time.time()

        # 设置实际并发数
        actual_max_workers = min(max_workers, total) if total else 1
        logger.info(f"[LLM判断] 开始对 {total} 条新日志进行大模型判断，并发数: {actual_max_workers}...")
        logger.info('-' * 80)

        # 定义单条日志处理函数
        def process_single_log(log_entry):
            """处理单条日志的LLM判断"""
            try:
                entry_id = str(log_entry['id'])
                question = log_entry['question']
                predicted = log_entry['predicted_text']
                answer = answer_map.get(entry_id, "")

                # 调用LLM判断
                llm_result = self.call_llm_judge_api(question, answer, predicted)

                if llm_result == "1":
                    return {'type': 'correct', 'entry': log_entry}
                else:
                    # 添加LLM判断结果到错误日志
                    error_entry = log_entry.copy()
                    error_entry['llm_judge'] = '0'
                    error_entry['ground_truth'] = answer
                    return {'type': 'error', 'entry': error_entry}

            except Exception as e:
                logger.error(f"[LLM判断] 处理日志ID {log_entry.get('id')} 失败: {e}")
                return {'type': 'error', 'entry': log_entry, 'error': str(e)}

        # 并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=actual_max_workers) as executor:
            future_to_log = {executor.submit(process_single_log, log): log for log in new_logs}

            for completed_count, future in enumerate(concurrent.futures.as_completed(future_to_log), start=1):
                try:
                    result = future.result()
                    if result['type'] == 'correct':
                        correct_logs.append(result['entry'])
                    else:
                        error_logs.append(result['entry'])
                except Exception as e:
                    error_logs.append(future_to_log[future])

                # 进度显示
                if completed_count % 10 == 0 or completed_count == total:
                    elapsed = time.time() - start_time
                    speed = completed_count / elapsed if elapsed > 0 else 0.0
                    progress = completed_count / total * 100 if total else 100.0
                    remaining = (total - completed_count) / speed if speed > 0 else 0

                    logger.info('')
                    logger.info(f'[LLM判断进度] {completed_count}/{total} ({progress:.1f}%)')
                    logger.info(f'   速度: {speed:.2f} 条/秒')
                    logger.info(f'   已用时: {elapsed:.1f} 秒')
                    logger.info(f'   预计剩余: {remaining:.1f} 秒')
                    logger.info(f'   正确: {len(correct_logs)} | 错误: {len(error_logs)}')
                    logger.info('-' * 80)

        # 保存错误日志到单独文件
        if error_logs:
            error_df = pd.DataFrame(error_logs)
            # 确保列顺序一致
            cols = ['id', 'question', 'retrieved_slices', 'correctness_score', 'predicted_text',
                    'ground_truth', 'llm_judge', 'timestamp']
            for col in cols:
                if col not in error_df.columns:
                    error_df[col] = ""
            error_df = error_df[cols]

            if os.path.exists(self.error_log_path):
                # 追加模式
                existing_df = pd.read_csv(self.error_log_path)
                error_df = pd.concat([existing_df, error_df], ignore_index=True)
                # 去重
                error_df = error_df.drop_duplicates(subset=['id'], keep='last')

            error_df.to_csv(self.error_log_path, index=False, encoding='utf-8')
            logger.info(f"[LLM判断] 错误日志已保存至: {self.error_log_path}，共 {len(error_logs)} 条")
        else:
            logger.info(f"[LLM判断] 无错误日志，无需保存")

        logger.info(f"[LLM判断] 完成！正确: {len(correct_logs)} | 错误: {len(error_logs)} | 正确率: {len(correct_logs) / total * 100 if total > 0 else 0:.2f}%")

        return correct_logs, error_logs

    def rejudge_existing_logs(self, test_df: pd.DataFrame):
        """
        一次性数据迁移：对现有日志进行LLM重新判断，过滤出正确的结果

        场景：现有 rag_feedback_logs.csv 是之前未经过LLM判断生成的，
        需要对其进行LLM判断过滤，只保留正确的条目。

        Args:
            test_df: 原始测试数据DataFrame（用于获取ground_truth）
        """
        if self.log_df.empty:
            logger.warning("[一次性重判] 当前日志为空，无需重判")
            return

        logger.info(f"[一次性重判] 开始对现有 {len(self.log_df)} 条日志进行LLM判断...")

        # 将现有日志转为dict列表，作为new_logs传入
        existing_logs = self.log_df.to_dict('records')
        for log in existing_logs:
            # 确保所有字段都是字符串或数值类型
            for key in log:
                if pd.isna(log[key]):
                    log[key] = ""

        # 调用LLM判断
        correct_logs, error_logs = self.llm_judge_and_filter(test_df, existing_logs)

        if not correct_logs:
            logger.warning("[一次性重判] 所有日志均被判定为错误，保留原数据")
            return

        # 用LLM判断正确的结果重建日志DataFrame
        logger.info(f"[一次性重判] 用 {len(correct_logs)} 条正确日志重建日志文件...")
        self.log_df = pd.DataFrame(correct_logs)
        
        # 确保列顺序一致
        cols = ['id', 'question', 'retrieved_slices', 'correctness_score', 'predicted_text', 'timestamp']
        for col in cols:
            if col not in self.log_df.columns:
                self.log_df[col] = ""
        self.log_df = self.log_df[cols]

        # 保存到CSV
        self.log_df.to_csv(self.log_path, index=False, encoding='utf-8')
        logger.info(f"[一次性重判] 日志文件已重建，共 {len(self.log_df)} 条正确记录")

        # 全量重建向量库
        logger.info("[一次性重判] 全量重建向量库...")
        self._reload_logs_to_vector_db()

        # 更新MD5记录
        try:
            new_md5 = get_file_md5_hex(self.log_path)
            if new_md5 and not self._is_md5_registered(new_md5):
                self._append_md5_record(new_md5)
                logger.info(f"[一次性重判] 新MD5已记录: {new_md5}")
        except Exception as e:
            logger.error(f"[一次性重判] 更新MD5失败: {e}")

        logger.info(f"[一次性重判] 完成！正确: {len(correct_logs)} | 错误: {len(error_logs)}")

    def _generate_unique_id(self, base_id: str) -> str:
        """
        生成唯一的向量库ID，避免重复
        格式：log_<base_id>_<时间戳哈希>
        """
        # 生成时间戳+base_id的哈希值
        unique_suffix = hashlib.md5(f"{base_id}_{datetime.now().timestamp()}".encode()).hexdigest()[:8]
        return f"log_{base_id}_{unique_suffix}"

    def _is_md5_registered(self, md5_for_check: str) -> bool:
        """
        参考标准方法：检查 MD5 哈希是否已存在于记录文件中
        """
        if not os.path.exists(self.md5_path):
            # 创建空文件
            open(self.md5_path, "w", encoding="utf-8").close()
            return False
            
        with open(self.md5_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if line.strip() == md5_for_check:
                    return True
        return False

    def _append_md5_record(self, md5_for_save: str):
        """
        参考标准方法：将新生成的 MD5 追加到存储文件末尾
        """
        with open(self.md5_path, "a", encoding="utf-8") as f:
            f.write(md5_for_save + "\n")
        logger.info("新 MD5 记录已追加至存储文件。")

    def _check_and_reload_vector_db(self):
        """检查 CSV 文件的记录数量，只有当记录增加时才增量添加新记录"""
        try:
            csv_record_count = len(self.log_df)
            vector_db_record_count = self._get_vector_db_record_count()
            
            logger.info(f"[增量检查] CSV记录数: {csv_record_count}, 向量库记录数: {vector_db_record_count}")
            
            if csv_record_count > vector_db_record_count:
                # 记录数量增加，增量添加新记录
                new_records_count = csv_record_count - vector_db_record_count
                logger.warning(f"[增量添加] 检测到有 {new_records_count} 条新记录，开始增量添加...")
                self._incrementally_add_new_records()
            else:
                logger.info("[增量检查] 向量库记录数已同步，无需添加新记录。")
                
        except Exception as e:
            logger.error("增量检查失败: %s", e, exc_info=True)
            # 发生错误时执行全量重载作为降级策略
            logger.warning("[降级策略] 增量检查失败，尝试全量重载...")
            self._reload_logs_to_vector_db()

    def _get_vector_db_record_count(self) -> int:
        """获取向量库中已加载的记录数量"""
        try:
            if self.logs_collection is None:
                return 0
            
            # 分页获取所有记录来计数（Chroma API 没有直接的 count() 方法）
            count = 0
            batch_size = 5000
            offset = 0
            while True:
                result = self.logs_collection.get(limit=batch_size, offset=offset)
                ids = result.get('ids', [])
                count += len(ids)
                if len(ids) < batch_size:
                    break
                offset += batch_size
            
            logger.info(f"[向量库统计] 当前记录数: {count}")
            return count
        except Exception as e:
            logger.warning(f"[向量库统计] 获取记录数失败: {e}，返回0")
            return 0

    def _incrementally_add_new_records(self):
        """增量添加新记录到向量库（不删除已有记录）"""
        if self.log_df.empty:
            logger.info("[增量添加] CSV日志为空，无需添加。")
            return
            
        try:
            # 获取向量库中已存在的记录数量
            existing_count = self._get_vector_db_record_count()
            
            # 获取新增的记录（从已有记录之后开始）
            if existing_count >= len(self.log_df):
                logger.info("[增量添加] 向量库记录数已超过CSV，无需添加新记录。")
                return
                
            new_records = self.log_df.iloc[existing_count:]
            logger.info(f"[增量添加] 准备添加 {len(new_records)} 条新记录...")
            
            texts = []
            metadatas = []
            ids = []
            
            for idx, (_, row) in enumerate(new_records.iterrows()):
                if pd.isna(row['question']) or str(row['id']).strip() == "":
                    continue
                
                doc_text = f"Question: {row['question']}\nRetrieved Slices: {row['retrieved_slices']}"
                texts.append(doc_text)
                metadatas.append({
                    "id": str(row['id']),
                    "question": str(row['question']),
                    "retrieved_slices": str(row['retrieved_slices']) if not pd.isna(row['retrieved_slices']) else "",
                    "correctness_score": str(row['correctness_score']) if not pd.isna(row['correctness_score']) else "0.0",
                    "predicted_text": str(row['predicted_text']) if not pd.isna(row['predicted_text']) else "",
                    "timestamp": str(row['timestamp']) if not pd.isna(row['timestamp']) else ""
                })
                unique_id = self._generate_unique_id(str(row['id']))
                ids.append(unique_id)
                
                if (idx + 1) % 1000 == 0:
                    logger.info(f"[增量添加] 已处理 {idx + 1}/{len(new_records)} 条新记录")
            
            logger.info(f"[增量添加] 文本数据构建完成，共 {len(texts)} 条")
            
            if texts:
                # 分批添加
                batch_size = 100
                total_batches = (len(texts) + batch_size - 1) // batch_size
                logger.info(f"[增量添加] 开始分批添加向量，共 {total_batches} 批")
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    batch_metadatas = metadatas[i:i+batch_size]
                    batch_ids = ids[i:i+batch_size]
                    try:
                        self.logs_collection.add_texts(
                            texts=batch_texts,
                            metadatas=batch_metadatas,
                            ids=batch_ids
                        )
                        logger.info(f"[增量添加] 已添加第 {i//batch_size + 1}/{total_batches} 批数据")
                    except Exception as e:
                        logger.error(f"[增量添加] 添加第 {i//batch_size + 1}/{total_batches} 批失败: {e}")
                        continue
                
                final_count = self._get_vector_db_record_count()
                logger.info(f"[增量添加] 完成！成功添加 {len(texts)} 条记录，向量库现有 {final_count} 条记录。")
            else:
                logger.info("[增量添加] 没有有效记录需要添加。")
                
        except Exception as e:
            logger.error(f"[增量添加] 失败: {e}", exc_info=True)
            # 降级：执行全量重载
            logger.warning("[增量添加] 降级为全量重载...")
            self._reload_logs_to_vector_db()

    def _reload_logs_to_vector_db(self):
        """清空原有的日志向量集合并重新全量加载 CSV 里的数据"""
        logger.info("[_reload_logs_to_vector_db] 开始执行重载...")
        
        if self.log_df.empty:
            logger.info("CSV 日志为空，无需同步至向量数据库。")
            return

        try:
            logger.info("[_reload_logs_to_vector_db] 步骤1: 删除整个向量库目录...")
            import shutil
            import gc
            
            if os.path.exists(self.persist_directory):
                try:
                    self.logs_collection._client.delete_collection(Config.collection_name)
                    logger.info("[_reload_logs_to_vector_db] 已通过 Chroma API 删除集合")
                except Exception as e:
                    logger.warning(f"[_reload_logs_to_vector_db] 通过 Chroma API 删除集合失败: {e}，尝试文件系统方式...")
                    del self.logs_collection
                    gc.collect()
                    shutil.rmtree(self.persist_directory)
                    logger.info("[_reload_logs_to_vector_db] 已通过文件系统删除旧的向量库目录")
            
            logger.info("[_reload_logs_to_vector_db] 步骤2: 重建向量库集合...")
            os.makedirs(self.persist_directory, exist_ok=True)
            self.logs_collection = Chroma(
                collection_name=Config.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
                collection_metadata={"hnsw:space": "cosine"}
            )
            logger.info("[_reload_logs_to_vector_db] 向量库集合已重建")
            
        except Exception as e:
            logger.error("重建向量库失败: %s", e, exc_info=True)
            return

        # 全量构建新文本并加入向量库
        texts = []
        metadatas = []
        ids = []

        logger.info("[_reload_logs_to_vector_db] 步骤3: 开始构建文本数据...")
        for idx, (_, row) in enumerate(self.log_df.iterrows()):
            if pd.isna(row['question']) or str(row['id']).strip() == "":
                continue
            
            doc_text = f"Question: {row['question']}\nRetrieved Slices: {row['retrieved_slices']}"
            texts.append(doc_text)
            metadatas.append({
                "id": str(row['id']),
                "question": str(row['question']),
                "retrieved_slices": str(row['retrieved_slices']) if not pd.isna(row['retrieved_slices']) else "",
                "correctness_score": str(row['correctness_score']) if not pd.isna(row['correctness_score']) else "0.0",
                "predicted_text": str(row['predicted_text']) if not pd.isna(row['predicted_text']) else "",
                "timestamp": str(row['timestamp']) if not pd.isna(row['timestamp']) else ""
            })
            # 关键修复3：生成唯一ID
            unique_id = self._generate_unique_id(str(row['id']))
            ids.append(unique_id)
            
            if (idx + 1) % 1000 == 0:
                logger.info(f"[_reload_logs_to_vector_db] 已处理 {idx + 1}/{len(self.log_df)} 条记录")
        
        logger.info(f"[_reload_logs_to_vector_db] 步骤4: 文本数据构建完成，共 {len(texts)} 条")

        if texts:
            # 分批添加（避免单次添加过多数据）
            batch_size = 100
            total_batches = (len(texts) + batch_size - 1) // batch_size
            logger.info(f"[_reload_logs_to_vector_db] 步骤5: 开始分批添加向量，共 {total_batches} 批")
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_metadatas = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                try:
                    logger.info(f"[_reload_logs_to_vector_db] 添加第 {i//batch_size + 1}/{total_batches} 批，共 {len(batch_texts)} 条...")
                    self.logs_collection.add_texts(
                        texts=batch_texts,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                    logger.info(f"已添加第 {i//batch_size + 1}/{total_batches} 批数据，共 {len(batch_texts)} 条")
                except Exception as e:
                    logger.error(f"添加第 {i//batch_size + 1}/{total_batches} 批数据失败: {e}", exc_info=True)
                    continue
            
            logger.info("重载完成，成功全量同步 %d 条记录至日志向量库。", len(texts))

    def extract_response_text(self, response):
        """
        递归提取响应文本，兼容不同模型返回格式
        （可能是字符串、Message对象、dict、list等）

        Args:
            response: 模型返回的各种格式响应

        Returns:
            str: 提取的纯文本内容
        """
        if isinstance(response, str):
            return response
        if hasattr(response, 'content'):
            return self.extract_response_text(response.content)
        if isinstance(response, dict):
            return self.extract_response_text(
                response.get('content') or response.get('text') or response.get('message') or ""
            )
        if isinstance(response, (list, tuple)) and response:
            return self.extract_response_text(response[0])
        return str(response)

    def evaluate_text_similarity(self, response: str, reference: str) -> float:
        """
        评估生成文本与参考答案的相似度
        采用 BLEU + 词汇重叠的混合评分方式

        Args:
            response: 生成文本
            reference: 参考答案

        Returns:
            float: 相似度得分 [0, 1]
        """
        # 文本归一化并分词
        reference_tokens = normalize_text(reference).split()
        response_tokens = normalize_text(response).split()

        # 防止空文本导致除零错误
        if not reference_tokens or not response_tokens:
            return 0.0

        # 使用BLEU分数（带平滑）
        smoothing = SmoothingFunction().method1
        bleu = sentence_bleu([reference_tokens], response_tokens, smoothing_function=smoothing)

        # 计算词汇重叠率
        overlap = len(set(reference_tokens) & set(response_tokens)) / max(len(set(reference_tokens)), 1)

        # 混合评分：BLEU权重 + 词汇重叠权重
        return float(min(1.0, Config.bleu_weight * bleu + Config.overlap_weight * overlap))

    def normalize_retrieval_results(self, retrieved_slices):
        """
        统一不同RAG引擎的检索结果格式
        确保返回统一的字典列表：{'id': ..., 'content': ...}

        Args:
            retrieved_slices: 原始检索结果（可能是各种格式）

        Returns:
            list: 标准化的结果列表
        """
        normalized = []
        for item in retrieved_slices:
            if isinstance(item, dict):
                # 从字典中提取id和内容
                doc_id = item.get('id') or item.get('slice_id') or item.get('metadata', {}).get('id')
                content = item.get('content') or item.get('text') or item.get('page_content') or ""
            else:
                # 从对象中提取属性
                doc_id = getattr(item, 'id', None) or getattr(item, 'slice_id', None)
                content = getattr(item, 'page_content', None) or getattr(item, 'content', None) or ""

                # 如果内容为空，尝试从metadata中获取
                if not content and hasattr(item, 'metadata'):
                    metadata = getattr(item, 'metadata')
                    if isinstance(metadata, dict):
                        content = metadata.get('content') or metadata.get('text') or ""

            normalized.append({
                'id': doc_id or str(item),
                'content': content or ""
            })
        return normalized

    def retrieve_slices(self, base_rag_engine, query, k=Config.retrieval_k):
        """
        适配不同RAG引擎的检索接口
        支持 search / similarity_search / search_documents / invoke 等多种方式

        Args:
            base_rag_engine: RAG引擎实例
            query: 查询问题
            k: 返回结果数量

        Returns:
            list: 检索到的切片列表
        """
        # 根据引擎支持的接口调用相应方法
        if hasattr(base_rag_engine, 'search'):
            results = base_rag_engine.search(query, k=k)
        elif hasattr(base_rag_engine, 'similarity_search'):
            results = base_rag_engine.similarity_search(query, k=k)
        elif hasattr(base_rag_engine, 'search_documents'):
            results = base_rag_engine.search_documents(query, k=k)
        elif hasattr(base_rag_engine, 'invoke'):
            results = base_rag_engine.invoke(query)
        else:
            raise AttributeError('base_rag_engine does not support search, similarity_search, search_documents or invoke')

        if results is None:
            return []
        if isinstance(results, dict):
            results = [results]
        return self.normalize_retrieval_results(results)

    def get_vlm_prediction(self, image_path, prompt, ground_truth=None):
        """
        调用多模态大模型（Llava）进行预测并评估

        Args:
            image_path: 图片路径
            prompt: 文本提示词
            ground_truth: 可选的参考答案

        Returns:
            tuple: (相似度得分, 生成文本)
        """
        # 构建完整提示词
        clean_prompt = f"{prompt}\n\n请结合图片内容和上下文回答问题。"
        if image_path and os.path.exists(image_path):
            image_hint = f"图片路径: {image_path}"
            full_prompt = f"{image_hint}\n{clean_prompt}"
        else:
            full_prompt = clean_prompt

        # 构建消息并调用模型
        message = HumanMessage(content=full_prompt)
        response = None
        if hasattr(self.multimodal_llm, 'invoke'):
            response = self.multimodal_llm.invoke([message])
        elif hasattr(self.multimodal_llm, 'generate'):
            response = self.multimodal_llm.generate([message])
        else:
            raise AttributeError('multimodal_llm does not support invoke or generate')

        # 提取文本并清理
        response_text = self.extract_response_text(response)
        response_text = response_text.replace("\r", " ").replace("\n", " ").strip()

        # 计算相似度得分
        score = self.evaluate_text_similarity(response_text, ground_truth) if ground_truth is not None else 0.0
        return score, response_text

    def evaluate_and_log(self, test_df, base_rag_engine, max_workers=50):
        """
        步骤1：执行测试并记录 CSV 日志（并行版本）
        对每条测试数据：
          - 用 base_rag_engine 检索相关切片
          - 用多模态模型生成回答
          - 用 BLEU 评估正确性
          - 结果写入日志文件

        Args:
            test_df: 测试数据DataFrame
            base_rag_engine: RAG引擎实例
            max_workers: 最大并发线程数，默认50
        """
        import concurrent.futures

        # 获取当前日志中已经成功测试过的 ID 集合
        existing_ids = set()
        if not self.log_df.empty and 'id' in self.log_df.columns:
            existing_ids = set(self.log_df['id'].dropna().astype(str).tolist())

        # 筛选出需要测试的数据（排除已存在的）
        test_rows = []
        for _, row in test_df.iterrows():
            current_id = str(row['id'])
            if current_id not in existing_ids:
                test_rows.append(row)
            else:
                logger.info('[增量跳过] 测试ID: %s 已经存在于日志中，跳过本次评估。', current_id)

        if not test_rows:
            logger.info('[增量检查完成] 所有测试数据均已存在于日志中，未触发新测试。')
            return

        # 定义单行处理函数
        def process_single_row(row):
            """处理单行测试数据"""
            try:
                # 测试输出：显示问题
                logger.info('%s', '*' * 50)
                logger.info('[测试ID: %s] 问题: %s', row['id'], row['question'])
                logger.info('%s', '*' * 50)

                # 调用VLM Agent进行预测
                score, response_text, slice_ids = self.call_vlm_agent(row['image'], row['question'], row.get('answer'))

                # 测试输出：显示预测结果和切片ID
                logger.info('%s', '*' * 50)
                logger.info('[测试ID: %s] 预测结果: %s', row['id'], response_text)
                logger.info('[测试ID: %s] 正确性得分: %.4f', row['id'], score)
                logger.info('[测试ID: %s] agent实际使用的切片ID: %s', row['id'], slice_ids)
                logger.info('%s', '*' * 50)

                # 返回日志条目
                log_entry = {
                    'id': row['id'],
                    'question': row['question'],
                    'retrieved_slices': "|".join(slice_ids),
                    'correctness_score': score,
                    'predicted_text': response_text,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                return log_entry

            except Exception as e:
                logger.error('[测试ID: %s] 处理失败: %s', row['id'], str(e), exc_info=True)
                return None

        # 并行处理
        new_logs = []
        total_tasks = len(test_rows)
        start_time = time.time()
        failed_count = 0

        logger.info('[并行处理] 开始测试，共 %d 条任务，并发数: %d', total_tasks, max_workers)
        logger.info('-' * 80)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_row = {executor.submit(process_single_row, row): row for row in test_rows}

            for completed_count, future in enumerate(concurrent.futures.as_completed(future_to_row), start=1):
                try:
                    log_entry = future.result()
                    if log_entry is not None:
                        new_logs.append(log_entry)
                    else:
                        failed_count += 1
                except Exception as e:
                    failed_count += 1
                    row = future_to_row[future]
                    logger.error('[ERROR] 测试ID: %s 执行出错: %s', row['id'], str(e))

                # 进度显示
                if completed_count % 10 == 0 or completed_count == total_tasks:
                    elapsed = time.time() - start_time
                    speed = completed_count / elapsed if elapsed > 0 else 0.0
                    progress = completed_count / total_tasks * 100 if total_tasks else 100.0
                    remaining = (total_tasks - completed_count) / speed if speed > 0 else 0

                    logger.info('')
                    logger.info('[PROGRESS] 进度更新 [%d/%d]', completed_count, total_tasks)
                    logger.info('   进度: %.1f%%', progress)
                    logger.info('   速度: %.2f 条/秒', speed)
                    logger.info('   已用时: %.1f 秒', elapsed)
                    logger.info('   预计剩余: %.1f 秒', remaining)
                    logger.info('   已完成: %d条有效记录 | 失败: %d条', len(new_logs), failed_count)
                    logger.info('-' * 80)

        # 4. 如果确实有新跑出来的数据，先进行LLM判断，再更新 CSV 和向量数据库
        if new_logs:
            # 4.1 先调用LLM对所有新日志进行判断，过滤出正确的和错误的
            correct_logs, error_logs = self.llm_judge_and_filter(test_df, new_logs)
            
            # 4.2 只将LLM判断正确的结果写入CSV和向量库
            if correct_logs:
                new_df = pd.DataFrame(correct_logs)
                self.log_df = pd.concat([self.log_df, new_df], ignore_index=True)
                
                # 去重后保存
                self._deduplicate_log_df()

                # 增量添加到向量库（使用唯一ID）
                for log_entry in correct_logs:
                    doc_text = f"Question: {log_entry['question']}\nRetrieved Slices: {log_entry['retrieved_slices']}"
                    unique_id = self._generate_unique_id(str(log_entry['id']))
                    self.logs_collection.add_texts(
                        texts=[doc_text],
                        metadatas=[{
                            "id": str(log_entry['id']),
                            "question": log_entry['question'],
                            "retrieved_slices": log_entry['retrieved_slices'],
                            "correctness_score": str(log_entry['correctness_score']),
                            "predicted_text": log_entry['predicted_text'],
                            "timestamp": log_entry['timestamp']
                        }],
                        ids=[unique_id]
                    )
                
                # 当由于新跑的数据写入导致 CSV 更新后，把最新的 MD5 写入记录文件，确保系统下一次运行顺利通过一致性检查
                try:
                    new_md5 = get_file_md5_hex(self.log_path)
                    if new_md5 and not self._is_md5_registered(new_md5):
                        self._append_md5_record(new_md5)
                except Exception as e:
                    logger.error(f"评估增量结束后更新新生成文件的 MD5 失败: {e}")

                logger.info(f"日志已更新（LLM判断正确），新增 {len(correct_logs)} 条记录，当前总条数: {len(self.log_df)}，已同步至向量数据库")
            else:
                logger.warning(f"[LLM判断] 所有 {len(new_logs)} 条新日志均被判定为错误，未写入任何记录至向量库")
            
            logger.info(f"[LLM判断汇总] 正确: {len(correct_logs)} | 错误: {len(error_logs)}")
        else:
            logger.info('[增量检查完成] 所有测试数据均已存在于日志中，未触发新测试。')

    def calculate_membership_degree(self, query, k=Config.membership_k, w1=Config.membership_w1, w2=Config.membership_w2, fit_threshold=0.5, top_p=3):
        """
        计算新问题与日志中相关内容的隶属度

        隶属度综合考虑：
        1. 新问题与日志问题+切片的相似度
        2. 日志中的正确性分数（作为该问题的可信度）

        Args:
            query: 新查询问题
            k: 检索相关日志条目的数量
            w1: 相似度权重
            w2: 正确性分数权重
            fit_threshold: 隶属度阈值，用于筛选合格日志
            top_p: 最多返回的合格隶属度数量

        Returns:
            dict: 包含以下字段的结果字典
                - membership_score: 综合隶属度得分 [0, 1]
                - top_logs: 前k条相关日志的详细信息
                - weighted_slices: 加权后的推荐切片（按隶属度排序）
                - qualified_log_count: 满足阈值的日志条数
                - qualified_memberships: 满足阈值的日志隶属度列表（按降序排列，最多top_p个）
        """
        # 验证权重和为1
        if abs(w1 + w2 - 1.0) > 1e-6:
            logger.warning('权重和不为1，进行自动归一化: w1=%.2f, w2=%.2f', w1, w2)
            total = w1 + w2
            w1 = w1 / total
            w2 = w2 / total

        # 1. 在日志库中检索相关条目
        try:
            results = self.logs_collection.similarity_search_with_relevance_scores(query, k=k)
        except Exception as e:
            logger.error('日志库检索失败: %s', e)
            return {
                'membership_score': 0.0,
                'top_logs': [],
                'weighted_slices': [],
                'qualified_log_count': 0,
                'qualified_memberships': []
            }

        if not results:
            logger.warning('未找到相关日志条目')
            return {
                'membership_score': 0.0,
                'top_logs': [],
                'weighted_slices': [],
                'qualified_log_count': 0,
                'qualified_memberships': []
            }

        # 2. 计算加权隶属度
        top_logs = []
        slice_memberships = {}
        total_membership = 0.0
        qualified_memberships = []
        all_memberships = []

        for doc, sim_score in results:
            # 从metadata中提取信息
            metadata = doc.metadata
            correctness = float(metadata.get('correctness_score', 0.0))
            retrieved_slices = metadata.get('retrieved_slices', '').split('|') if metadata.get('retrieved_slices') else []

            # 计算综合隶属度：mu = w1 * similarity + w2 * correctness
            membership = (w1 * sim_score) + (w2 * correctness)
            total_membership += membership

            # 记录所有隶属度
            all_memberships.append(membership)

            # 记录日志条目
            log_info = {
                'id': metadata.get('id', 'unknown'),
                'question': metadata.get('question', ''),
                'similarity': float(sim_score),
                'correctness_score': correctness,
                'membership_degree': membership,
                'retrieved_slices': retrieved_slices
            }
            top_logs.append(log_info)

            # 3. 仅基于合格日志统计切片的隶属度（取最高隶属度）
            if membership >= fit_threshold:
                qualified_memberships.append(membership)
                for slice_id in retrieved_slices:
                    if slice_id:
                        if slice_id not in slice_memberships or membership > slice_memberships[slice_id]:
                            slice_memberships[slice_id] = membership

        # 平均隶属度
        avg_membership = total_membership / len(top_logs) if top_logs else 0.0
        
        # 最大隶属度（所有检索日志中的最大值）
        max_membership = max(all_memberships) if all_memberships else 0.0

        # 5. 对合格隶属度按降序排序，最多保留 top_p 个
        qualified_memberships.sort(reverse=True)
        qualified_memberships = qualified_memberships[:top_p]

        # 6. 生成加权切片列表（按隶属度排序）
        weighted_slices = [
            {
                'slice_id': slice_id,
                'membership_degree': membership,
                'normalized_membership': membership / max_membership if max_membership > 0 else 0.0
            }
            for slice_id, membership in sorted(slice_memberships.items(),
                                                key=lambda x: x[1], reverse=True)
        ]

        logger.info('隶属度计算完成: 平均得分=%.4f, 最大得分=%.4f, 相关日志=%d条, 合格日志=%d条, 推荐切片=%d个',
                   avg_membership, max_membership, len(top_logs), len(qualified_memberships), len(weighted_slices))

        return {
            'membership_score': avg_membership,
            'max_membership': max_membership,
            'top_logs': top_logs,
            'weighted_slices': weighted_slices,
            'qualified_log_count': len(qualified_memberships),
            'qualified_memberships': qualified_memberships
        }


class RealRAGEngine:
    """真实的 RAG 引擎包装器，使用 RscsvBuilder 的切片向量库"""

    def __init__(self):
        # 使用RscsvBuilder构建RAG系统
        self.builder = RscsvBuilder()
        self.slice_collection = self.builder.slice_collection
        
        # 配置参数（与 RscsvServiceRscsv 保持一致）
        self.slice_k = SLICE_K
        self.top_p = TOP_P


    def search(self, query, k=None):
        """从向量库中检索相关切片（与 RscsvServiceRscsv.hybrid_retrieve 逻辑保持一致）

        Args:
            query: 查询问题
            k: 返回结果数量（已废弃，使用配置中的 slice_k）

        Returns:
            list: 检索到的切片列表（按相似度降序排序后取前 top_p 条）
        """
        # 使用 similarity_search_with_relevance_scores 获取带相似度得分的结果
        slice_results = self.slice_collection.similarity_search_with_relevance_scores(query, k=self.slice_k)
        
        if not slice_results:
            return []
            
        # 按相似度得分降序排序（与 RscsvServiceRscsv.hybrid_retrieve 保持一致）
        sorted_results = sorted(slice_results, key=lambda x: x[1], reverse=True)
        
        # 取前 top_p 条结果（与 RscsvServiceRscsv.hybrid_retrieve 保持一致）
        top_results = sorted_results[:self.top_p]
        
        results = []
        for doc, score in top_results:
            slice_id = getattr(doc, 'id', None) or doc.metadata.get('slice_id')
            
            if not slice_id:
                # 如果都没有（防呆兜底），才使用 hash
                slice_id = str(hash(doc.page_content))
                
            content = doc.page_content
            results.append({
                'id': slice_id,
                'content': content,
                'similarity_score': float(score)
            })
        return results


def run_real_test():
    """
    运行真实数据的测试
    
    运行模式通过 RUN_MODE 变量控制：
    - "incremental": 增量更新模式（默认）- 加载已有日志，MD5校验后决定是否重载向量库
    - "force_reload": 强制全量重载模式 - 清空CSV和向量库，重新开始评估
    - "rejudge": 重判模式 - 清除向量库，重新将rag_feedback_logs.csv数据入库
    """
    
    logger.info(f'=== 开始 SARSemanticCacheSystem 真实数据测试 [模式: {RUN_MODE}] ===')
    
    # 根据运行模式设置参数
    force_full_reload = (RUN_MODE == "force_reload")
    clear_vector_db_only = (RUN_MODE == "rejudge")
    
    # 创建系统实例
    system = SARSemanticCacheSystem(
        force_full_reload=force_full_reload,
        clear_vector_db_only=clear_vector_db_only
    )
    
    # rejudge 模式：清除向量库后直接将 CSV 数据入库，然后结束
    if RUN_MODE == "rejudge":
        logger.info('--- [重判模式] 清除向量库并重新入库 rag_feedback_logs.csv ---')
        # 注意：clear_vector_db_only=True 时，已在 __init__ 中处理了向量库清空
        # 此时 system.log_df 已加载 CSV 数据，只需调用 _reload_logs_to_vector_db
        if not system.log_df.empty:
            system._reload_logs_to_vector_db()
            logger.info(f'[重判模式] 成功入库 {len(system.log_df)} 条日志记录')
        else:
            logger.warning('[重判模式] rag_feedback_logs.csv 为空，无需入库')
        logger.info('=== 重判模式完成 ===')
        return
    
    # 创建真实的 RAG 引擎
    base_rag_engine = RealRAGEngine()
    
    # 读取测试数据
    test_csv_path = "../../dataset_split/test.csv"  # 相对于 agent/rag/ 的路径
    if not os.path.exists(test_csv_path):
        test_csv_path = "../dataset_split/test.csv"
    if not os.path.exists(test_csv_path):
        test_csv_path = "dataset_split/test.csv"
    
    if not os.path.exists(test_csv_path):
        logger.error('错误：找不到测试文件 %s', test_csv_path)
        return
    
    # 读取测试数据
    test_df = pd.read_csv(test_csv_path, nrows=Config.test_nrows)
    
    # 测试数据检查
    logger.info('测试数据加载成功，文件路径: %s', test_csv_path)
    for i, row in test_df.iterrows():
        logger.info('  测试ID: %s, 问题: %s, 图片: %s', row['id'], row['question'], row['image'])
    
    # 修正图片路径（确保完整路径）
    test_df['image'] = test_df['image'].apply(lambda x: os.path.join(Config.base_image_dir, x.lstrip('/')))
    
    logger.info('加载了 %d 条测试数据', len(test_df))
    
    # 步骤1: 评估和记录
    logger.info('--- 步骤1: 执行评估和记录 ---')
    try:
        system.evaluate_and_log(test_df, base_rag_engine)
    except Exception as e:
        logger.exception('步骤1出错: %s', e)
        return
    
    # 显示日志
    logger.info('当前日志条数: %d', len(system.log_df))
    if len(system.log_df) > 0:
        logger.info('最后3条日志:')
        for i, row in system.log_df.tail(3).iterrows():
            logger.info('  日志 %d: ID=%s, 得分=%.4f', i+1, row['id'], row['correctness_score'])
    
    # 步骤2: 隆属度计算测试
    logger.info('--- 步骤2: 隆属度计算测试 ---')
    if len(system.log_df) > 0:
        try:
            query = test_df.iloc[0]['question']
            membership_result = system.calculate_membership_degree(
                query,
                k=Config.membership_k,
                w1=Config.membership_w1,
                w2=Config.membership_w2
            )
            
            logger.info('隶属度计算结果:')
            logger.info('  综合隶属度得分: %.4f', membership_result['membership_score'])
            logger.info('  相关日志数: %d', len(membership_result['top_logs']))
            
            if membership_result['top_logs']:
                logger.info('  前2条相关日志:')
                for log in membership_result['top_logs'][:2]:
                    logger.info('    ID=%s, 相似度=%.4f, 正确性=%.4f, 隶属度=%.4f',
                               log['id'], log['similarity'], log['correctness_score'], log['membership_degree'])
            
            if membership_result['weighted_slices']:
                logger.info('  推荐切片（前3个）:')
                for slice_info in membership_result['weighted_slices'][:3]:
                    logger.info('    slice_id=%s, 隆属度=%.4f',
                               slice_info['slice_id'], slice_info['membership_degree'])
        except Exception as e:
            logger.exception('步骤2出错: %s', e)
    
    logger.info('=== 测试完成 ===')


if __name__ == "__main__":
    # 直接运行，不再使用命令行参数，通过 RUN_MODE 变量控制模式
    run_real_test()