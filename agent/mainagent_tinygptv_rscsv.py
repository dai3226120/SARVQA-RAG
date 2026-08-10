"""TinyGPT-V RSCSV Agent 变体（对应 benchmark MODEL_KEY = "agent-text-tinygptv_rscsv"）

复用 InternVL RSCSV Agent 的完整流程（doubao 收集 RAG-RSCSV 切片检索 → 视觉模型回答），
仅两处不同：
1. 最终视觉模型替换为 tinygptv_model
2. RAG 上下文按字数截断 —— tinygptv 服务端 max_model_len=2048（RSCSV 模式 slice_k 条
   切片全量拼进上下文，不截断则视觉请求必 400）

改名：发布正式模型时把 tinygptv 替换为正式模型名（类名/文件名/vision_model 引用）。
"""
import os
import sys

# 路径注入必须早于任何项目内 import（与 mainagent_internVL_rscsv.py 一致）
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)  # 项目根目录
for p in (current_dir, root_dir):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import base64
import re

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from model.factory import tinygptv_model
from utils.prompt_loader import load_system_prompts
from utils.logger_handler import logger
from mainagent_internVL_rscsv import MainAgent as _InternVLRscsvMainAgent

# 同 mainagent_tinygptv：服务端 max_model_len=2048，系统提示+RAG 合计截断到 1200 字符
MAX_SYSTEM_TEXT_CHARS = 1200


class MainAgent(_InternVLRscsvMainAgent):
    """两步式 agent：doubao 收集 RAG（RSCSV 切片检索）→ tinygptv 多模态最终回答"""

    def __init__(self):
        super().__init__()
        self.vision_model = tinygptv_model

    def execute_stream(self, query: str, image_file=None):
        """同基类流程，唯一差异：RAG 上下文截断（服务端 2048 上下文限制）"""
        # 准备最终视觉模型要用的多模态内容（包含图片）
        multi_modal_content = [{"type": "text", "text": query}]
        if image_file is not None:
            try:
                if hasattr(image_file, "seek"):
                    image_file.seek(0)
                image_bytes = image_file.read()
                image_type = getattr(image_file, "type", "image/jpeg") or "image/jpeg"
                base64_image = base64.b64encode(image_bytes).decode("utf-8")
                multi_modal_content.append(
                    {"type": "image_url", "image_url": {"url": f"data:{image_type};base64,{base64_image}"}}
                )
            except Exception as e:
                logger.error(f"[MainAgent] 读取图像数据失败: {e}", exc_info=True)
                yield "无法读取上传图像，请重试。"
                return

        self.last_slice_ids = []

        # ----- 第一步：用 doubao agent 收集 RAG 信息（仅文本） -----
        try:
            agent_input = {"messages": [HumanMessage(content=query)]}
            result = self.agent.invoke(agent_input)
            messages = result["messages"]
        except Exception as e:
            logger.error(f"[MainAgent] Agent 调用失败: {e}", exc_info=True)
            yield f"执行出错：{str(e)}"
            return

        # 提取所有工具返回的文本，并从中提取切片ID
        rag_parts = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                content = msg.content
                if isinstance(content, list):
                    rag_parts.append(" ".join(str(part) for part in content))
                else:
                    rag_parts.append(str(content))

                # 从工具返回内容中提取切片ID
                if msg.name == "rag_rscsv_rscsv":
                    tool_content = str(content)
                    match = re.search(r'<!-- SLICE_IDS: (.*?) -->', tool_content)
                    if match:
                        ids_str = match.group(1).strip()
                        if ids_str:
                            self.last_slice_ids = ids_str.split(',')

        rag_context = "\n\n".join(rag_parts)

        # 保存到实例属性供 get_rag_output() 使用
        self._last_rag_context = rag_context

        # ----- 第二步：用 tinygptv 多模态模型生成最终答案 -----
        system_text = load_system_prompts()
        if rag_context:
            system_text += f"\n\n以下是检索到的相关背景信息：\n{rag_context}"

        # ---- tinygptv 特有：系统提示+RAG 合计截断（服务端上下文 2048，输入超限必 400）----
        if len(system_text) > MAX_SYSTEM_TEXT_CHARS:
            system_text = system_text[:MAX_SYSTEM_TEXT_CHARS]

        # 追加最终答案的格式要求（放在截断之后，保证格式指令完整）
        system_text += "\n\n最终答案用一句话英文说明，不超过150字，不要使用例如或括号。"

        vision_messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=multi_modal_content)
        ]

        try:
            for chunk in self.vision_model.stream(vision_messages):
                if hasattr(chunk, "content") and chunk.content:
                    if isinstance(chunk.content, str):
                        yield chunk.content
                    elif isinstance(chunk.content, list):
                        yield "".join([item.get("text", "") for item in chunk.content if isinstance(item, dict)])
        except Exception as e:
            logger.error(f"[MainAgent] 视觉模型生成失败: {e}", exc_info=True)
            yield f"视觉模型出错：{str(e)}"
