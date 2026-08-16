import os
import sys

# 路径注入必须早于任何项目内 import：utils/tools/rag/model 都在项目根目录，
# 而 streamlit run / python agent/mainagent.py 都只会把 agent 目录放进 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)  # 项目根目录（agent 的父目录）
for p in (current_dir, root_dir):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import base64
import re
# 导入LangChain 相关模块
from langchain.agents import create_agent
from deepagents import create_deep_agent
from langchain_community.chat_models import ChatTongyi
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
# 导入项目内的模块
from model.factory import chat_model, embed_model, huggingface_embed_model, doubao_seed_20_mini_model, internvl2_8b_model, internvl3_5_8b_model
from utils.prompt_loader import load_system_prompts
from tools.agent_tools import rag_summarize,get_weather,get_user_location, get_user_id, rag_knowledge_context, to_openai_tools, _get_service
from tools.middleware import monitor_tool,log_before_model,report_prompt_switch,calculate_hit_rate
from utils.logger_handler import logger

class MainAgent:
    def __init__(self, model=None, tools=None):
        """Args:
            model: 聊天模型实例（None 时用默认 doubao_seed_20_mini_model）
            tools: 工具列表（None 时用默认 [rag_knowledge_context]，仅阶段0 知识库上下文）
        """
        self.tools = tools if tools is not None else [rag_knowledge_context]
        self.model = model
        self.rag_output = ""  # 累积 RAG 工具输出文本
        self.last_slice_ids = []

        self.agent = create_agent(
            model=self.model if self.model is not None else doubao_seed_20_mini_model,
            tools=self.tools,
            system_prompt=load_system_prompts(),
            middleware=[log_before_model, monitor_tool]
        )

    def rebuild_agent(self, model):
        """切换模型：只重建 agent 图，Chroma 存储不重建（省时）"""
        self.model = model
        self.agent = create_agent(
            model=model,
            tools=self.tools,
            system_prompt=load_system_prompts(),
            middleware=[log_before_model, monitor_tool]
        )

    def _build_messages(self, query: str, image_file=None, history=None) -> list:
        """构造输入消息：历史文本（纯文本）+ 最新一轮（含可选图片）"""
        messages = []
        for h in history or []:
            content = h.get("content", "")
            if h.get("role") == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))

        content = [{"type": "text", "text": query}]
        if image_file is not None:
            try:
                if hasattr(image_file, "seek"):
                    image_file.seek(0)
                image_bytes = image_file.read()
                image_type = getattr(image_file, "type", "image/jpeg") or "image/jpeg"
                base64_image = base64.b64encode(image_bytes).decode("utf-8")
                content.append(
                    {"type": "image_url", "image_url": {"url": f"data:{image_type};base64,{base64_image}"}}
                )
            except Exception as e:
                logger.error(f"[MainAgent] 读取图像数据失败: {e}", exc_info=True)
                raise ValueError("无法读取上传图像，请重试。")
        messages.append(HumanMessage(content=content))
        return messages

    def get_last_trace(self):
        """获取最后一次 RAG 检索的过程记录（知识库上下文检索，来自 KnowledgeRagService）"""
        return _get_service('rag').get_last_trace()

    # 多模态输入版本
    def execute_stream(self, query: str, image_file=None, history=None):
        """流式执行 agent 问答

        Args:
            query: 本轮问题
            image_file: 当前图片（file-like，可带 .type 属性）
            history: 之前轮次 [{"role": "user"|"assistant", "content": str}, ...]
        """
        try:
            input_dict = {"messages": self._build_messages(query, image_file, history)}
        except ValueError as e:
            yield str(e)
            return
        yielded_ai_text_len = 0
        self.rag_output = ""  # 重置 RAG 输出
        rag_parts = []

        try:
            for chunk in self.agent.stream(input_dict, stream_mode="values"):
                if "messages" not in chunk or not chunk["messages"]:
                    continue
                
                messages = chunk["messages"]
                last_msg = messages[-1]

                # 调试日志
                print(f"\n[DEBUG] Message type: {last_msg.type}", file=sys.stderr)
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    print(f"[DEBUG] Tool calls: {last_msg.tool_calls}", file=sys.stderr)
                if hasattr(last_msg, "name") and last_msg.name:
                    print(f"[DEBUG] Tool response: {last_msg.name}", file=sys.stderr)

                # 从工具返回消息中提取切片ID（检索工具输出含 SLICE_IDS 标记时生效）
                if last_msg.type == "tool" and hasattr(last_msg, "content"):
                    tool_content = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
                    match = re.search(r'<!-- SLICE_IDS: (.*?) -->', tool_content)
                    if match:
                        ids_str = match.group(1).strip()
                        if ids_str:
                            self.last_slice_ids = ids_str.split(',')
                            print(f"[DEBUG] Extracted slice_ids: {self.last_slice_ids}", file=sys.stderr)

                # 捕获 RAG 工具输出
                if last_msg.type == "tool" and hasattr(last_msg, "content"):
                    tc = last_msg.content
                    if isinstance(tc, list):
                        rag_parts.append(" ".join(str(part) for part in tc))
                    else:
                        rag_parts.append(str(tc))

                if last_msg.type == "ai" and last_msg.content:
                    # 只有没有tool_calls时才是最终回答
                    if not getattr(last_msg, "tool_calls", None):
                        full_content = last_msg.content
                        
                        current_text = ""
                        if isinstance(full_content, str):
                            current_text = full_content
                        elif isinstance(full_content, list):
                            current_text = "".join([
                                item.get("text", "") for item in full_content 
                                if isinstance(item, dict) and item.get("type") == "text"
                            ])
                        
                        if len(current_text) > yielded_ai_text_len:
                            new_chunk = current_text[yielded_ai_text_len:]
                            yield new_chunk
                            yielded_ai_text_len = len(current_text)
        except Exception as e:
            logger.error(f"[MainAgent] 模型调用失败: {e}", exc_info=True)
            yield f"执行出错：{str(e)}"
            return
        finally:
            self.rag_output = "\n\n".join(rag_parts) if rag_parts else ""

    def get_rag_output(self) -> str:
        """获取最后一次 RAG 工具调用的累积输出文本"""
        return getattr(self, 'rag_output', '')

    def get_tool_hit_stats(self):
        """获取工具调用命中率统计（仅阶段0 知识库上下文，无隶属度命中概念，返回空）"""
        return {}

if __name__=="__main__":
    agent = MainAgent()

    # 匹配隶属度库
    # image_path = "C:\\dataset\\SAR-TEXT\\SAR-TEXT-data\\Image\\SEN12\\ROIs1158_spring_s1_10_p124.png"
    image_path = "C:\\dataset\\SAR-TEXT\\SAR-TEXT-data\\Image\\QXSLAB_SAROPT\\7756.png"
    # image_path = "C:\\dataset\\SAR-TEXT\\SAR-TEXT-data\\Image\\whu-sar-opt\\NH49E010014_2_3.tif"
    # image_path = "C:\\dataset\\SAR-TEXT\\SAR-TEXT-data\\Image\\whu-sar-opt\\NH49E010014_2_3.tif"
    # image_path = ""
    # query = "What is the scale of human development relative to the natural landscape?"
    query = "Is there evidence of water bodies, like ponds or streams, in this landscape?"
    # query = "How does the SAR image depict the transition between forested and non-forested areas?"
    # query = "How does the urban area appear in this SAR image, and what distinguishes it from the surrounding landscape?"
    image_file = None
    
    try:
        # 尝试打开图片
        if image_path: 
            image_file = open(image_path, "rb")
            
        for chunk in agent.execute_stream(query, image_file=image_file):
            print(chunk, end="", flush=True)
            
    except FileNotFoundError:
        print(f"⚠️ 告警：图片文件 {image_path} 不存在，将仅使用文本进行查询...\n")
        try:
            for chunk in agent.execute_stream(query, image_file=None):
                print(chunk, end="", flush=True)
        except Exception as e:
             print(f"\n纯文本模式执行出错：{e}")
             
    except Exception as e:
        print(f"\n执行出错：{e}")
        
    finally:
        # 如果文件被成功打开过，确保关闭它
        if image_file and not getattr(image_file, "closed", True):
            image_file.close()
        print()  # 结束后换行