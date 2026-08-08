"""
VLM 评估模块
负责 VLM 预测调用 + 文本相似度评估（BLEU + 词汇重叠）
源自原 rag_rsfit_builder.py 中：
  - _init_vlm_agent / call_vlm_agent / get_vlm_prediction
  - extract_response_text / evaluate_text_similarity
  - normalize_retrieval_results / retrieve_slices
"""
import os
import re
import threading

from langchain_core.messages import HumanMessage

from rag.core.config import rag_config
from model.factory import huggingface_embed_model, doubao_seed_20_mini_model
from utils.data_process import image_to_base64, normalize_text
from utils.logger_handler import logger


class VlmEvaluator:
    """
    VLM（视觉语言模型）评估器
    职责：调用 VLM Agent 获取切片 → 调用多模态模型预测 → 评估文本相似度
    """

    # 类级共享的切片检索服务（懒初始化 + 双检锁）：
    # 每个实例首次检索会独立构建一份 ~757MB 的全量 faiss 精确索引，
    # init_membership_logs 以 100 并发调用时若每 worker 各自 new 实例，
    # 内存按并发线性增长（100 并发 ≈ 75GB，必 OOM）。共享单例使索引全局仅一份。
    _shared_rscsv_service = None
    _rscsv_service_lock = threading.Lock()

    def _get_rscsv_service(self):
        """获取类级共享的 SliceRetrievalService（线程安全懒初始化）"""
        if VlmEvaluator._shared_rscsv_service is None:
            with VlmEvaluator._rscsv_service_lock:
                if VlmEvaluator._shared_rscsv_service is None:
                    from rag.services.slice_service import SliceRetrievalService
                    VlmEvaluator._shared_rscsv_service = SliceRetrievalService()
        return VlmEvaluator._shared_rscsv_service

    def __init__(self, multimodal_llm=None):
        """
        Args:
            multimodal_llm: 多模态大模型实例，默认使用 doubao_seed_20_mini_model
        """
        self._multimodal_llm = multimodal_llm or doubao_seed_20_mini_model
        self._vlm_agent = None
        self._vlm_agent_initialized = False
        self._vlm_agent_type = rag_config.vlm_agent_type
        logger.info(f"[VlmEvaluator] 配置类型: {self._vlm_agent_type}（懒加载）")

    def _init_vlm_agent(self):
        """根据配置初始化 VLM Agent（懒加载，避免循环导入）"""
        try:
            import importlib
            import sys

            agent_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "agent",
            )
            if agent_dir not in sys.path:
                sys.path.insert(0, agent_dir)

            if self._vlm_agent_type.lower() == "internvl":
                module = importlib.import_module("mainagent_internVL_rscsv")
            else:
                module = importlib.import_module("mainagent_rscsv")

            MainAgent = getattr(module, "MainAgent")
            self._vlm_agent = MainAgent()
            logger.info(f"[VlmEvaluator] 已初始化 VLM Agent: {self._vlm_agent_type}")
        except Exception as e:
            logger.error(f"[VlmEvaluator] VLM Agent 初始化失败: {e}", exc_info=True)
            self._vlm_agent = None

    def call_vlm_agent(self, image_path, question, ground_truth=None):
        """
        调用 VLM Agent 获取预测结果

        Args:
            image_path: 图片路径
            question: 问题文本
            ground_truth: 参考答案

        Returns:
            tuple: (score, response_text, slice_ids)
        """
        # 阶段1：调用共享切片检索服务获取切片（复用单例，避免每 worker 独立构建 757MB 索引）
        rscsv_service = self._get_rscsv_service()
        rscsv_result = rscsv_service.retrieve(question)

        # 从返回结果中提取切片 ID
        slice_ids_match = re.search(r"<!-- SLICE_IDS: (.*?) -->", rscsv_result)
        slice_ids = []
        if slice_ids_match:
            ids_str = slice_ids_match.group(1).strip()
            if ids_str:
                slice_ids = ids_str.split(",")
        logger.info(f"[VlmEvaluator] 获取的切片ID: {slice_ids}")

        if not image_path or not os.path.exists(image_path):
            logger.warning(f"[VlmEvaluator] 图片不存在: {image_path}")
            return 0.0, "", slice_ids

        try:
            # 阶段2：构建包含切片上下文的完整提示词
            enhanced_prompt = (
                f"{rscsv_result}\n\n请结合以上参考资料和图片内容回答问题：{question}"
            )

            # 阶段3：调用多模态模型进行预测
            score, response_text = self.get_vlm_prediction(
                image_path, enhanced_prompt, ground_truth
            )

        except Exception as e:
            logger.error(f"[VlmEvaluator] 调用失败: {e}", exc_info=True)
            score, response_text = self.get_vlm_prediction(
                image_path, question, ground_truth
            )

        return score, response_text, slice_ids

    def get_vlm_prediction(self, image_path, prompt, ground_truth=None):
        """
        调用多模态大模型进行预测并评估

        Args:
            image_path: 图片路径
            prompt: 文本提示词
            ground_truth: 可选的参考答案

        Returns:
            tuple: (相似度得分, 生成文本)
        """
        clean_prompt = f"{prompt}\n\n请结合图片内容和上下文回答问题。"
        if image_path and os.path.exists(image_path):
            full_prompt = f"图片路径: {image_path}\n{clean_prompt}"
        else:
            full_prompt = clean_prompt

        message = HumanMessage(content=full_prompt)
        response = None
        if hasattr(self._multimodal_llm, "invoke"):
            response = self._multimodal_llm.invoke([message])
        elif hasattr(self._multimodal_llm, "generate"):
            response = self._multimodal_llm.generate([message])
        else:
            raise AttributeError(
                "multimodal_llm does not support invoke or generate"
            )

        response_text = self._extract_response_text(response)
        response_text = response_text.replace("\r", " ").replace("\n", " ").strip()

        score = (
            self.evaluate_text_similarity(response_text, ground_truth)
            if ground_truth is not None
            else 0.0
        )
        return score, response_text

    @staticmethod
    def _extract_response_text(response) -> str:
        """递归提取响应文本，兼容不同模型返回格式"""
        if isinstance(response, str):
            return response
        if hasattr(response, "content"):
            return VlmEvaluator._extract_response_text(response.content)
        if isinstance(response, dict):
            return VlmEvaluator._extract_response_text(
                response.get("content")
                or response.get("text")
                or response.get("message")
                or ""
            )
        if isinstance(response, (list, tuple)) and response:
            return VlmEvaluator._extract_response_text(response[0])
        return str(response)

    @staticmethod
    def evaluate_text_similarity(response: str, reference: str) -> float:
        """
        评估生成文本与参考答案的语义相似度

        采用与 benchmark/core/metrics.py calculate_cosine_similarity 一致的方案：
        bge-m3 嵌入两个文本 → sklearn 余弦相似度（语义级，同义改写不再归零）。

        旧实现（已弃用，仅作历史参考）：
        BLEU + 词汇重叠的混合评分 —— 对 normalize_text 分词后的文本计算
        nltk sentence_bleu(method1 平滑) 与 词袋重叠率，
        score = min(1.0, bleu_weight*bleu + overlap_weight*overlap)。
        该实现是词法级表面匹配，同义改写即归零，故替换为语义余弦。

        Args:
            response: 生成文本
            reference: 参考答案

        Returns:
            float: 余弦相似度得分 [0, 1]
        """
        if not response or not reference:
            return 0.0
        try:
            from model.factory import huggingface_embed_model
            from sklearn.metrics.pairwise import cosine_similarity

            emb = huggingface_embed_model.embed_documents([response, reference])
            return round(float(cosine_similarity([emb[0]], [emb[1]])[0][0]), 4)
        except Exception as e:
            logger.error(f"[VlmEvaluator] 余弦相似度计算失败: {e}")
            return 0.0
