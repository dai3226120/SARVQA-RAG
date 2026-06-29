# LangSmith 监控和日志工具
from dotenv import load_dotenv
# load_dotenv()

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import base64
# 导入LangChain 相关模块
from langchain.agents import create_agent
from deepagents import create_deep_agent
from langchain_community.chat_models import ChatTongyi
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
# 导入项目内的模块
from model.factory import chat_model, embed_model, huggingface_embed_model, doubao_seed_20_mini_model, internvl2_8b_model, internvl3_5_8b_model
from utils.prompt_loader import load_system_prompts
from tools.agent_tools import rag_summarize,get_weather,get_user_location, get_user_id, rag_rscsv, to_openai_tools, rag_rscsv_rscsv
from tools.middleware import monitor_tool,log_before_model,report_prompt_switch
from utils.logger_handler import logger

class MainAgent:
    def __init__(self):

        # self.tools = [rag_summarize, rag_rscsv]
        self.tools = [rag_rscsv_rscsv]
        self.last_slice_ids = []

        self.agent = create_agent(
            model = doubao_seed_20_mini_model,  # 使用多模态模型
            # model = chat_model,  # 使用通义千问聊天模型
            # tools = [get_user_location, get_weather, rag_summarize, rag_rscsv],
            tools = self.tools,
            system_prompt=load_system_prompts(),
            middleware=[log_before_model,monitor_tool]
        )

        # 2. internvl 作为多模态视觉最终回答模型
        self.vision_model = internvl3_5_8b_model
        # self.vision_model = internvl2_8b_model
        
    # 多模态输入版本
    def execute_stream(self, query: str, image_file=None):
        import re
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
            # agent 只收文本，避免浪费视觉推理
            agent_input = {"messages": [HumanMessage(content=query)]}
            # 使用 invoke 获取完整消息历史
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
                    # 有些工具可能返回分段内容，统一转为字符串
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

        # ----- 第二步：用 internvl 多模态模型生成最终答案 -----
        system_text = load_system_prompts()
        if rag_context:
            system_text += f"\n\n以下是检索到的相关背景信息：\n{rag_context}"

        # 追加最终答案的格式要求
        system_text += "\n\n最终答案用一句话英文说明，不超过150字，不要使用例如或括号。"

        vision_messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=multi_modal_content)
        ]

        try:
            for chunk in self.vision_model.stream(vision_messages):
                # 处理不同chunk的content类型
                if hasattr(chunk, "content") and chunk.content:
                    if isinstance(chunk.content, str):
                        yield chunk.content
                    elif isinstance(chunk.content, list):
                        # 极少数模型可能返回list，按文本拼接
                        yield "".join([item.get("text", "") for item in chunk.content if isinstance(item, dict)])
        except Exception as e:
            logger.error(f"[MainAgent] 视觉模型生成失败: {e}", exc_info=True)
            yield f"视觉模型出错：{str(e)}"

if __name__=="__main__":
    agent = MainAgent()

    # 匹配隶属度库
    # image_path = "C:\\dataset\\SAR-TEXT\\SAR-TEXT-data\\Image\\SEN12\\ROIs1158_spring_s1_10_p124.png"
    image_path = "C:\\dataset\\SAR-TEXT\\SAR-TEXT-data\\Image\\QXSLAB_SAROPT\\7756.png"
    # image_path = ""
    # query = "What is the scale of human development relative to the natural landscape?"
    query = "Is there evidence of water bodies, like ponds or streams, in this landscape?"
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