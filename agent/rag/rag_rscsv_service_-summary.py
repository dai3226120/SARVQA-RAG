import os
from rag.rag_rscsv_builder import RscsvBuilder
from rag.rag_rsfit_builder import SARSemanticCacheSystem  # 引入隶属度计算系统
from utils.config_handler import chroma_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


class RscsvService:
    _membership_total_calls = 0
    _membership_hit_calls = 0

    def __init__(self):
        # 1. 基础构建器与切片集合
        self.builder = RscsvBuilder()
        self.slice_collection = self.builder.slice_collection
        self.fit_threshold = float(chroma_conf["retrieval"].get("fit_threshold", 0.5))
        self.slice_k = int(chroma_conf.get("slice_k", 10))
        self.membership_k = int(chroma_conf.get("membership_k", 5))
        self.top_p = int(chroma_conf.get("top_p", 3))
        
        # 2. 初始化缓存系统（用于隶属度计算）
        self.cache_system = SARSemanticCacheSystem()

    @classmethod
    def reset_membership_stats(cls):
        cls._membership_total_calls = 0
        cls._membership_hit_calls = 0

    @classmethod
    def get_membership_hit_rate_static(cls):
        if cls._membership_total_calls == 0:
            return 0.0
        return cls._membership_hit_calls / cls._membership_total_calls

    def hybrid_retrieve(self, query: str, slice_k: int = 10, membership_k: int = 10, top_p: int =3 , fit_threshold: float = 0.5) -> str:
        # ==========================================
        # 阶段 1: 优先调用隶属度计算 (缓存拦截与校验)
        # ==========================================
        membership_hit = False
        try:
            RscsvService._membership_total_calls += 1
            
            membership_result = self.cache_system.calculate_membership_degree(query)
            membership_score = membership_result.get('membership_score', 0.0)
            
            if membership_score >= fit_threshold and membership_result.get('weighted_slices'):
                logger.info(f"【隶属度命中】得分: {membership_score:.4f} >= 阈值: {fit_threshold}")
                
                # 从缓存结果中提取前 k 个切片 ID
                top_slice_ids = [
                    s['slice_id'] for s in membership_result['weighted_slices'][:membership_k]
                ]
                logger.info(f" 正在尝试从切片库获取以下 ID 的数据: {top_slice_ids}")
                
                slices_data = self.slice_collection.get(ids=top_slice_ids)
                documents = slices_data.get('documents', [])

                logger.info(f" 切片库实际返回了 {len(documents)} 条文档内容")
                
                if documents:
                    membership_hit = True
                    content = "\n---\n".join(documents)
                    return (
                        f"【匹配隶属度缓存】综合隶属度得分={membership_score:.4f}  \n{content}"
                    )
            else:
                logger.info(f"【隶属度未命中/未达标】得分: {membership_score:.4f} < 阈值: {fit_threshold}")

        except Exception as e:
            logger.error(f"隶属度计算过程发生异常，降级到基础检索: {str(e)}")
        finally:
            if membership_hit:
                RscsvService._membership_hit_calls += 1

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
        return self.hybrid_retrieve(query, slice_k=self.slice_k, membership_k=self.membership_k, top_p=self.top_p, fit_threshold=self.fit_threshold)


if __name__ == "__main__":
    service = RscsvService()
    print(service.retrieve("Does the radar image show any weather-related disturbances, like storms or rain?"))