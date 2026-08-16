# -*- coding: utf-8 -*-
"""调试脚本：验证 create_agent 的 system_prompt 是否真的注入到发给模型的 messages 中"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)  # 项目根目录
agent_dir = os.path.join(root_dir, "agent")
for p in (agent_dir, root_dir):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest
from langchain_core.messages import HumanMessage
from model.factory import doubao_seed_20_mini_model
from tools.agent_tools import rag_knowledge_context
from utils.prompt_loader import load_system_prompts
from tools.middleware import log_before_model, monitor_tool


@wrap_model_call
def dump_messages(request: ModelRequest, handler):
    """打印每次模型调用真正收到的 messages"""
    print("=" * 70)
    print("[dump_messages] 模型调用 #%d, 消息数=%d" % (getattr(dump_messages, "_n", 0) + 1, len(request.messages)))
    if request.system_message is not None:
        print("  system_message 存在, 内容前300字:")
        print("   ", repr(str(request.system_message.content)[:300]))
    else:
        print("  system_message 为 None！")
    for i, m in enumerate(request.messages[:4]):
        print("  msg[%d] %-15s: %s" % (i, type(m).__name__, repr(str(m.content)[:150])))
    dump_messages._n = getattr(dump_messages, "_n", 0) + 1
    return handler(request)


def main():
    print(">>> 构造 agent（mainagent_internVL_knowledge 同款：doubao + rag_knowledge_context + system_prompt.txt）")
    agent = create_agent(
        model=doubao_seed_20_mini_model,
        tools=[rag_knowledge_context],
        system_prompt=load_system_prompts(),
        middleware=[log_before_model, monitor_tool, dump_messages],
    )
    query = "Is there any evidence of water bodies like ponds or streams in this SAR landscape?"
    print(">>> invoke:", query)
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    final = result["messages"][-1]
    print("=" * 70)
    print(">>> 最终回复 (%s):" % type(final).__name__)
    print(str(final.content)[:400])


if __name__ == "__main__":
    main()
