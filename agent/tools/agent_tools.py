# 导入必要的工具类和库
from langchain_core.tools import tool
from rag.services.knowledge_service import KnowledgeRagService  # 向量检索服务
from rag.services.membership_service import MembershipHybridService   # 隶属度混合检索服务
from rag.services.slice_service import SliceRetrievalService   # 切片检索服务
import random
import threading
from utils.config_handler import agent_conf  # 配置处理器
from utils.path_tool import get_abs_path    # 获取绝对路径工具
from utils.logger_handler import logger      # 日志处理器
import os

# 服务实例懒加载（PEP 562 模块级 __getattr__）：
# knowledge 模式（mainagent_knowledge）只注册 rag_knowledge_context 工具（阶段0 知识库检索），
# 若在模块顶部直接实例化三个服务，membership（切片库+类别中心预热）与 slice（faiss 全量索引）
# 也会被一并初始化——改为首次访问时才实例化，无关模式不加载无关服务。
_service_instances: dict = {}
_service_lock = threading.Lock()


def _create_service(name: str):
    """按需创建服务实例"""
    if name == "rag":
        return KnowledgeRagService()
    if name == "rag_rscsv_service":
        return MembershipHybridService()
    if name == "rag_rscsv_service_rscsv":
        return SliceRetrievalService()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _get_service(name: str):
    """按需获取服务实例（懒加载，线程安全；与模块级 __getattr__ 共用同一缓存）"""
    if name not in _service_instances:
        with _service_lock:
            if name not in _service_instances:
                _service_instances[name] = _create_service(name)
    return _service_instances[name]


def __getattr__(name: str):
    """模块级懒加载：rag / rag_rscsv_service / rag_rscsv_service_rscsv 首次访问时实例化
    （覆盖模块属性访问与 from-import；函数体内的全局名字查找不经过本函数，
    工具函数内部须用 _get_service 显式获取）"""
    if name in ("rag", "rag_rscsv_service", "rag_rscsv_service_rscsv"):
        return _get_service(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# 测试数据：用户ID列表和月份列表
user_ids = ["1001", "1002"]
month_arr = ["2025-01", "2025-02"]

# 全局外部数据缓存字典
external_data = {}

# --- 工具函数定义 ---
# 注意：函数体内的服务实例须用 _get_service 显式获取（懒加载），
# 不能直接写全局名 rag / rag_rscsv_service / rag_rscsv_service_rscsv
# （函数内全局名字查找不经过模块级 __getattr__，会 NameError）
@tool(description="从向量存储中检索遥感通用知识相关参考资料")
def rag_summarize(query: str) -> str:
    """遥感通用知识相关检索工具"""
    return _get_service('rag').rag_summarize(query)

@tool(description="从知识库中检索遥感通用知识参考上下文（仅阶段0检索，不做LLM总结）。注意：请务必直接透传用户的原始输入，不要做任何总结或改写。")
def rag_knowledge_context(query: str) -> str:
    """遥感知识库上下文检索工具（阶段0，仅检索不总结）"""
    print(f"[rag_knowledge_context tool] Received query: {query}")
    return _get_service('rag').retrieve_context(query)

# @tool(description="从向量存储中检索遥感问答相关参考资料")
@tool(description="从向量存储中检索遥感问答相关参考资料。注意：请务必直接透传用户的原始输入，不要做任何总结或改写。")
def rag_rscsv(query: str) -> str:
    """遥感问答检索工具"""
    print(f"[rag_rscsv tool] Received query: {query}")  # 调试日志，查看输入查询
    return _get_service('rag_rscsv_service').retrieve(query)

# @tool(description="从向量存储中检索遥感问答相关参考资料")
@tool(description="从向量存储中检索遥感问答相关参考资料。注意：请务必直接透传用户的原始输入，不要做任何总结或改写。")
def rag_rscsv_rscsv(query: str) -> str:
    """遥感问答检索工具"""
    print(f"[rag_rscsv_rscsv tool] Received query: {query}")  # 调试日志，查看输入查询
    return _get_service('rag_rscsv_service_rscsv').retrieve(query)

@tool(description="获取指定的城市天气，以消息字符串形式返回")
def get_weather(city: str) -> str:
    """模拟天气查询工具"""
    return f"城市{city}天气晴，无风"

@tool(description="获取所在地区城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    """随机返回城市名称"""
    return random.choice(["合肥", "包头", "兖州"])

@tool(description="获取用户ID，以纯字符串形式返回")
def get_user_id() -> str:
    """随机返回测试用户ID"""
    return random.choice(user_ids)

# --- 数据加载模块 ---
def generate_external_data():
    """加载外部数据到内存缓存"""
    global external_data  # 声明全局变量
    
    # 仅当数据未加载时执行
    if not external_data:
        # 获取配置中的数据文件路径
        external_data_path = get_abs_path(agent_conf["external_data_path"])
        
        # 检查文件是否存在
        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据不存在{external_data_path}")
        
        # 读取CSV文件（跳过标题行）
        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                # 解析CSV行数据
                arr: list[str] = line.strip().split(",")
                user_id: str = arr[0].replace('"', "")
                feature: str = arr[1].replace('"', "")
                efficiency: str = arr[2].replace('"', "")
                consumables: str = arr[3].replace('"', "")
                comparison: str = arr[4].replace('"', "")
                time: str = arr[5].replace('"', "")
                
                # 构建嵌套数据结构
                if user_id not in external_data:
                    external_data[user_id] = {}
                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }

# --- 业务工具函数 ---
@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    """随机返回测试月份"""
    return random.choice(month_arr)

@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回，如果未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    """查询用户指定月份的使用记录"""
    generate_external_data()  # 确保数据已加载
    
    try:
        # 尝试从缓存中获取数据
        return external_data[user_id][month]
    except KeyError:
        # 记录未找到数据的警告
        logger.warning(f"[fetch_external_data]未能检索到用户{user_id}在{month}的使用记录")
        return ""

@tool(description="无参数和返回值，调用后触发中间件，自动生成报告注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    """上下文注入工具（占位实现）"""
    return "fill_context_for_report已调用"



# ========== 🔑 新增：转换为 OpenAI 格式（vLLM 专用）==========
def to_openai_tools(langchain_tools):
    """将 LangChain 工具转换为 OpenAI/vLLM 格式"""
    openai_tools = []
    for tool in langchain_tools:
        # 从 LangChain tool 对象提取信息
        tool_func = tool.func
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": f"{tool.description} 的查询参数"
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False
                }
            }
        })
    return openai_tools

# 导出两种格式
__all__ = ['rag_summarize', 'rag_rscsv', 'rag_rscsv_rscsv', 'to_openai_tools']

# --- 主程序入口 ---
if __name__ == '__main__':
    # 测试数据查询功能
    print(fetch_external_data("1001", "2025-01"))
