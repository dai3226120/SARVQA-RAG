import time
import threading
from unittest import result
from sqlalchemy.orm import dynamic
from utils.prompt_loader import load_system_prompts,load_report_prompts,load_rag_prompts
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, before_agent, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from typing import Callable, Dict, Any
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.runtime import  Runtime
from utils.logger_handler import logger


class ToolLatencyTracker:
    _global_total_latency = 0.0
    _global_call_count = 0
    _lock = threading.Lock()
    _thread_local = threading.local()

    @classmethod
    def _get_thread_data(cls):
        if not hasattr(cls._thread_local, 'session_total_latency'):
            cls._thread_local.session_total_latency = 0.0
            cls._thread_local.session_call_count = 0
        return cls._thread_local

    @classmethod
    def record_retrieval_latency(cls, latency: float):
        thread_data = cls._get_thread_data()
        thread_data.session_total_latency += latency
        thread_data.session_call_count += 1
        with cls._lock:
            cls._global_total_latency += latency
            cls._global_call_count += 1

    @classmethod
    def get_session_retrieval_latency(cls) -> float:
        thread_data = cls._get_thread_data()
        return thread_data.session_total_latency

    @classmethod
    def get_session_retrieval_call_count(cls) -> int:
        thread_data = cls._get_thread_data()
        return thread_data.session_call_count

    @classmethod
    def reset_session(cls):
        thread_data = cls._get_thread_data()
        thread_data.session_total_latency = 0.0
        thread_data.session_call_count = 0

    @classmethod
    def get_global_total_latency(cls) -> float:
        with cls._lock:
            return cls._global_total_latency

    @classmethod
    def get_global_call_count(cls) -> int:
        with cls._lock:
            return cls._global_call_count

    @classmethod
    def get_global_avg_latency(cls) -> float:
        with cls._lock:
            if cls._global_call_count == 0:
                return 0.0
            return cls._global_total_latency / cls._global_call_count

    @classmethod
    def reset_global(cls):
        with cls._lock:
            cls._global_total_latency = 0.0
            cls._global_call_count = 0

def calculate_hit_rate(tool_name: str = None) -> Dict[str, float]:
    """
    计算隶属度命中率（从 RscsvService 获取统计数据）
    :param tool_name: 指定工具名（None则计算隶属度命中率）
    :return: 命中率字典，key为工具名，value为命中率（0-1）
    """
    hit_rates = {}
    
    try:
        from rag.rag_rscsv_service import RscsvService
        
        service = RscsvService()
        stats = service.get_membership_stats()
        hit_rates["rag_rscsv_membership"] = stats["hit_rate"]
        logger.info(f"隶属度命中率统计: 总调用={stats['total_calls']}, 命中={stats['hit_calls']}, 命中率={stats['hit_rate']:.4f}")
    except Exception as e:
        logger.error(f"获取隶属度命中率失败: {str(e)}")
        hit_rates["rag_rscsv_membership"] = 0.0
    
    return hit_rates

@wrap_tool_call
def monitor_tool(
        #qingqiu 数据封装
        request: ToolCallRequest,
        #执行函数本身
        handler:Callable[[ToolCallRequest],ToolMessage | Command],
) -> ToolMessage | Command:
    tool_name = request.tool_call['name']
    logger.info(f"[tool monitor]执行工具:{tool_name}")
    logger.info(f"[tool monitor]传入参数:{request.tool_call['args']}")

    is_retrieval_tool = tool_name in ("rag_rscsv", "rag_rscsv_rscsv")
    start_time = time.time() if is_retrieval_tool else None

    try:
       result = handler(request)
       logger.info(f"[tool monitor]工具:{tool_name}调用成功")

       if is_retrieval_tool and start_time is not None:
           latency = time.time() - start_time
           ToolLatencyTracker.record_retrieval_latency(latency)
           logger.info(f"[tool monitor]检索工具:{tool_name} 耗时:{latency:.4f}秒 本次会话累计:{ToolLatencyTracker.get_session_retrieval_latency():.4f}秒")

       if tool_name == "fill_context_for_report":
           request.runtime.context["report"] = True

       return result
    except Exception as e:
        logger.error(f"工具{tool_name}调用失败，原因:{str(e)}")
        raise e

@before_model
def log_before_model(
        state: AgentState,   #整个agent智能体中的状态记录
        runtime: Runtime,    #记录了整个执行过程中的上下文信息
):
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息)")
    # logger.info(f"[log_before_model]即将调用模型，内容：  \n{state['messages'][0].content}\n---------------------------  \n\n")
    # 获取最后一条消息的内容
    last_msg = state['messages'][-1]
    content = last_msg.content
    
    # 兼容性处理：判断 content 的类型
    if isinstance(content, str):
        # 如果是普通文本，直接 strip
        display_content = content.strip()
    elif isinstance(content, list):
        # 如果是多模态列表，提取出文本部分进行日志展示
        text_parts = [
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
            if isinstance(item, dict) and item.get("type") == "text" or not isinstance(item, dict)
        ]
        # 合并文本并截断，防止日志过长，同时加上 [Multimodal] 标记
        combined_text = "".join(text_parts).strip()
        display_content = f"[Multimodal Content] {combined_text}"
    else:
        display_content = str(content)

    logger.debug(f"[log_before_model] {type(last_msg).__name__} | {display_content}")
    # logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__}|{state['messages'][-1].content.strip()}")
    return None



@dynamic_prompt  #每一次提示词生成之前，调用此函数
def report_prompt_switch(request:ModelRequest):
    is_report = request.runtime.context.get("report", False)
    if is_report:  #是报告生成场景，返回报告生成提示词内容,这个地方想实现动态提示词切换，时间有限没有实现
        return load_report_prompts()
    return load_system_prompts()