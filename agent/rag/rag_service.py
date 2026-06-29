#总结：搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import doubao_seed_20_mini_model
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser


def print_prompt(prompt):#就是交给大模型前打印一下提示词
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt

class RagSummarizeService(object):
    def __init__(self):
        self.vertor_store = VectorStoreService()
        self.retriever = self.vertor_store.get_retriever() #向量检索器
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = doubao_seed_20_mini_model
        self.chain = self.__init__chain()

    def __init__chain(self):#chain的实现
        # chain = self.prompt_template | print_prompt |self.model | StrOutputParser()
        chain = self.prompt_template |self.model | StrOutputParser()
        return chain
    def retriever_docs(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)

    def rag_summarize(self, query: str) -> str:

        context_docs = self.retriever_docs(query)
        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1 #拿到资料片段拼接成字符串
            context += f"【参考资料{counter}】:参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        return self.chain.invoke(
            {
                "input": query,
                "context": context
            }
        )

if __name__ == "__main__":
    rag = RagSummarizeService()
    print(rag.rag_summarize("sar 图像"))