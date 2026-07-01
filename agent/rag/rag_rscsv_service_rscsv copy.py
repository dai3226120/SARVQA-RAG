import os
from rag.rag_rscsv_builder import RscsvBuilder
from utils.config_handler import chroma_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


class RscsvServiceRscsv:
    def __init__(self):
        # 1. 基础构建器与切片集合
        self.builder = RscsvBuilder()
        self.slice_collection = self.builder.slice_collection
        self.slice_k = int(chroma_conf.get("slice_k", 10))
        self.membership_k = int(chroma_conf.get("membership_k", 5))

    def hybrid_retrieve(self, query: str, slice_k: int = 10, membership_k: int = 10, top_p: int =3 , fit_threshold: float = 0.5) -> str:
        # ==========================================
        # 阶段 2: 隶属度不符合阈值，降级回退到基础切片检索
        # ==========================================
        slice_results = self.slice_collection.similarity_search(query, k=slice_k)
        if slice_results:
            # 拼接基础检索返回的多条内容
            content = "\n---\n".join([doc.page_content for doc in slice_results])
            return (
                f"【匹配基础切片】  \n{content}"
            )

        return "未检索到相关遥感问答参考资料。"

    def retrieve(self, query: str) -> str:
        # 确保配置中的 k 为整数
        k_val = int(chroma_conf.get("k", 1))
        return self.hybrid_retrieve(query, k=k_val)


if __name__ == "__main__":
    service = RscsvServiceRscsv()
    print(service.retrieve("Does the radar image show any weather-related disturbances, like storms or rain?"))