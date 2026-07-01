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
from langchain_core.messages import HumanMessage
# 导入项目内的模块
from model.factory import chat_model, embed_model, huggingface_embed_model, doubao_seed_20_mini_model, internvl2_8b_model, internvl3_5_8b_model
from utils.prompt_loader import load_system_prompts
from tools.agent_tools import rag_summarize,get_weather,get_user_location, get_user_id, rag_rscsv, to_openai_tools
from tools.middleware import monitor_tool,log_before_model,report_prompt_switch,calculate_hit_rate
from utils.logger_handler import logger

class MainAgent:
    def __init__(self):

        # self.tools = [rag_summarize, rag_rscsv]
        self.tools = [rag_rscsv]
        # self.tools = [rag_summarize]

        self.agent = create_agent(
            model = doubao_seed_20_mini_model,  # 使用多模态模型
            # model = chat_model,  # 使用通义千问聊天模型
            # tools = [get_user_location, get_weather, rag_summarize, rag_rscsv],
            tools = self.tools,
            system_prompt=load_system_prompts(),
            middleware=[log_before_model,monitor_tool]
        )
        
    # 多模态输入版本
    def execute_stream(self, query: str, image_file=None):
        content = [{"type": "text", "text": query}]

        if image_file is not None:
            try:
                if hasattr(image_file, "seek"):
                    image_file.seek(0)
                image_bytes = image_file.read()
                image_type = getattr(image_file, "type", "image/jpeg") or "image/jpeg"
                base64_image = base64.b64encode(image_bytes).decode("utf-8")
                content = [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": f"data:{image_type};base64,{base64_image}"}}
                ]
            except Exception as e:
                logger.error(f"[MainAgent] 读取图像数据失败: {e}", exc_info=True)
                yield "无法读取上传图像，请重试。"
                return

        input_dict = {"messages": [HumanMessage(content=content)]}
        yielded_ai_text_len = 0

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
            logger.error(f"[MainAgent] 模型调用失败: {e}", exc_info=True)  # 打印完整错误堆栈
            yield f"执行出错：{str(e)}"  # 返回具体错误，而非仅500
            return

    def get_tool_hit_stats(self):
        """获取工具调用命中率统计，返回隶属度命中率"""
        from rag.rag_rscsv_service import RscsvService
        stats = RscsvService.get_membership_stats_static()
        return {"rag_rscsv": stats["hit_rate"]}

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