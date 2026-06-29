#SummarizationMiddleware就是一个中间件，可以经之前的提问进行总结，
from langchain_openai import OpenAI, ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import InMemorySaver
from typing import Literal  #枚举
from langchain.agents.middleware import SummarizationMiddleware
from model.factory import chat_model

#openai没有讲通义千问纳入环境变量调用
#model = init_chat_model(model="deepseek-chat")

chekpointer = InMemorySaver()

agent1 = create_agent(
    model="deepseek-chat",

    middleware=[
        SummarizationMiddleware(
            model=chat_model,
            trigger=("messages", 3),
            keep=("messages", 1)
        )
    ],
    checkpointer=chekpointer,
)

config = {"configurable":{"thread_id":"thread_3"}}#设置用户id

agent1.invoke({"messages": [HumanMessage(content="你好我公司名叫中科博涯科技公司，我们公司是开发人工智能软件的公司")]},config)
agent1.invoke({"messages": [HumanMessage(content="我们公司在北京")]},config)
agent1.invoke({"messages": [HumanMessage(content="我们公司在主要做办公智能体和管理智能体软件")]},config)

final_response = agent1.invoke({"messages": [HumanMessage(content="你还记得我们公司叫什么名字吗？")]},config)
print(final_response)