import os
from rag.rag_rscsv_builder import RscsvBuilder
from rag.rag_rsfit_builder import SARSemanticCacheSystem  # 引入隶属度计算系统
from rag.rag_service import RagSummarizeService
from utils.config_handler import chroma_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


class RscsvService:
    # 类级别静态变量，所有实例共享统计数据
    _membership_total_calls = 0
    _membership_hit_calls = 0
    
    def __init__(self):
        # 1. 基础构建器与切片集合
        self.builder = RscsvBuilder()
        self.slice_collection = self.builder.slice_collection
        self.fit_threshold = float(chroma_conf["retrieval"].get("fit_threshold", 0.5))
        
        # 2. 初始化缓存系统（用于隶属度计算）
        self.cache_system = SARSemanticCacheSystem()
        
        # 3. 初始化 RAG 总结服务
        self.rag_summarize_service = RagSummarizeService()
        
        # 4. 配置参数
        self.top_p = int(chroma_conf.get("top_p", 3))
        self.slice_k = int(chroma_conf.get("slice_k", 10))
        self.membership_k = int(chroma_conf.get("membership_k", 5))

    def hybrid_retrieve(self, query: str, slice_k: int = 10, membership_k: int = 10, top_p: int =3 , fit_threshold: float = 0.5) -> str:
        # ==========================================
        # 阶段 0: RAG 向量检索
        # ==========================================
        try:
            context_docs = self.rag_summarize_service.retriever_docs(query)
            rag_context_parts = []
            for idx, doc in enumerate(context_docs, 1):
                rag_context_parts.append(f"【参考资料{idx}】:{doc.page_content}")
            rag_context = "\n".join(rag_context_parts)
            logger.info(f"【RAG检索】已完成向量检索，共获取{len(context_docs)}条参考资料")
        except Exception as e:
            logger.error(f"RAG检索过程发生异常，跳过RAG检索: {str(e)}")
            rag_context = ""

        # ==========================================
        # 阶段 1: 优先调用隶属度计算 (缓存拦截与校验)
        # ==========================================
        RscsvService._membership_total_calls += 1
        try:
            # 使用 membership_k 进行隶属度检索，传入阈值和 top_p
            membership_result = self.cache_system.calculate_membership_degree(
                query, k=membership_k, fit_threshold=fit_threshold, top_p=top_p
            )
            max_membership = membership_result.get('max_membership', 0.0)
            qualified_log_count = membership_result.get('qualified_log_count', 0)
            qualified_memberships = membership_result.get('qualified_memberships', [])
            
            # 构造隶属度列表字符串（最多展示 top_p 个）
            memberships_str = ", ".join([f"{m:.4f}" for m in qualified_memberships])
            
            # 判断是否有合格日志且存在推荐切片
            if qualified_log_count > 0 and membership_result.get('weighted_slices'):
                logger.info(f"【隶属度命中】合格日志: {qualified_log_count}条，隶属度(从大到小): [{memberships_str}]")
                
                # 获取隶属度最高的前 top_p 个切片ID（weighted_slices 已按隶属度降序排列）
                top_slice_info = membership_result['weighted_slices'][:top_p]
                top_slice_ids = [s['slice_id'] for s in top_slice_info]
                logger.info(f" 正在尝试从切片库获取以下 ID 的数据: {top_slice_ids}")
                
                # 使用 Chroma 的 get 方法通过 id 精准获取切片内容
                slices_data = self.slice_collection.get(ids=top_slice_ids)
                documents = slices_data.get('documents', [])
                metadatas = slices_data.get('metadatas', [])

                logger.info(f" 切片库实际返回了 {len(documents)} 条文档内容")
                
                if documents:
                    RscsvService._membership_hit_calls += 1
                    # 按隶属度得分重新排序文档（确保顺序一致）
                    doc_with_membership = []
                    for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
                        slice_id = metadata.get('slice_id') if isinstance(metadata, dict) else top_slice_ids[i]
                        # 查找对应的隶属度
                        membership_degree = next(
                            (s['membership_degree'] for s in top_slice_info if s['slice_id'] == slice_id),
                            0.0
                        )
                        doc_with_membership.append((doc, membership_degree))
                    
                    # 按隶属度降序排序
                    doc_with_membership.sort(key=lambda x: x[1], reverse=True)
                    
                    # 拼接提取出的切片内容（包含隶属度得分）
                    content = "\n---\n".join(
                        [f"隶属度得分: {score:.4f}\n{doc}" for doc, score in doc_with_membership]
                    )
                    rscsv_result = f"【匹配隶属度缓存】合格日志={qualified_log_count}条，隶属度(从大到小): [{memberships_str}] (共检索{membership_k}条，按隶属度排序后保留{len(doc_with_membership)}条)  \n{content}"
                    if rag_context:
                        return f"【RAG检索参考】\n{rag_context}\n\n==============================\n\n{rscsv_result}"
                    return rscsv_result
            else:
                logger.info(f"【隶属度未命中/未达标】合格日志: {qualified_log_count}条，最大隶属度: {max_membership:.4f}")

        except Exception as e:
            logger.error(f"隶属度计算过程发生异常，降级到基础检索: {str(e)}")

        # ==========================================
        # 阶段 2: 隶属度不符合阈值，降级回退到基础切片检索
        # ==========================================
        slice_results = self.slice_collection.similarity_search_with_relevance_scores(query, k=slice_k)
        if slice_results:
            # 按相似度得分降序排序
            sorted_results = sorted(slice_results, key=lambda x: x[1], reverse=True)
            
            # 取 top_p 条结果
            top_results = sorted_results[:top_p]
            
            # 拼接检索结果（包含相似度得分）
            content = "\n---\n".join(
                [f"相似度得分: {score:.4f}\n{doc.page_content}" for doc, score in top_results]
            )
            rscsv_result = f"【匹配基础切片】(共检索{slice_k}条，按相似度排序后保留{len(top_results)}条)  \n{content}"
            if rag_context:
                return f"【RAG检索参考】\n{rag_context}\n\n==============================\n\n{rscsv_result}"
            return rscsv_result

        if rag_context:
            return f"【RAG检索参考】\n{rag_context}\n\n==============================\n\n未检索到相关遥感问答参考资料。"
        return "未检索到相关遥感问答参考资料。"

    def retrieve(self, query: str) -> str:
        # 确保配置中的 k 为整数
        return self.hybrid_retrieve(query, slice_k=self.slice_k, membership_k=self.membership_k, top_p=self.top_p)
    
    def get_membership_hit_rate(self) -> float:
        """获取隶属度命中率"""
        if RscsvService._membership_total_calls == 0:
            return 0.0
        return RscsvService._membership_hit_calls / RscsvService._membership_total_calls
    
    @classmethod
    def get_membership_hit_rate_static(cls) -> float:
        """静态方法：获取隶属度命中率"""
        if cls._membership_total_calls == 0:
            return 0.0
        return cls._membership_hit_calls / cls._membership_total_calls
    
    def get_membership_stats(self) -> dict:
        """获取隶属度统计信息"""
        return {
            "total_calls": RscsvService._membership_total_calls,
            "hit_calls": RscsvService._membership_hit_calls,
            "hit_rate": self.get_membership_hit_rate()
        }
    
    @classmethod
    def get_membership_stats_static(cls) -> dict:
        """静态方法：获取隶属度统计信息"""
        return {
            "total_calls": cls._membership_total_calls,
            "hit_calls": cls._membership_hit_calls,
            "hit_rate": cls.get_membership_hit_rate_static()
        }
    
    @classmethod
    def reset_membership_stats(cls):
        """重置隶属度统计数据"""
        cls._membership_total_calls = 0
        cls._membership_hit_calls = 0


if __name__ == "__main__":
    service = RscsvService()
    print(service.retrieve("Does the radar image show any weather-related disturbances, like storms or rain?"))


    # print(service.retrieve("What is the scale of human development relative to the natural landscape?"))