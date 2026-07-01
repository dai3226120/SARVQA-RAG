import os
import pandas as pd
import numpy as np
from k_means_constrained import KMeansConstrained  # 带约束的KMeans聚类
from sklearn.metrics.pairwise import cosine_similarity  # 余弦相似度计算
from tqdm import tqdm  # 进度条显示
from langchain_chroma import Chroma  # LangChain的Chroma向量数据库集成
from model.factory import huggingface_embed_model  # 从模型工厂获取嵌入模型
from utils.config_handler import chroma_conf  # 配置加载
from utils.file_handler import get_file_md5_hex  # 文件MD5计算工具
from utils.path_tool import get_abs_path  # 路径处理工具
from utils.logger_handler import logger

class RscsvBuilder:
    """SAR数据的RAG构建器，负责数据预处理、聚类、切片生成和向量库构建"""
    
    def __init__(self):
        # 配置路径初始化
        self.embeddings = huggingface_embed_model  # 轻量级文本嵌入模型
        self.collection_name=chroma_conf["collection_name"]
        self.persist_directory=get_abs_path(chroma_conf["persist_diretory"])
        self.data_dir = get_abs_path(chroma_conf["data_path"])  # 数据存储目录
        self.sar_csv_path = get_abs_path("data/SAR-VQA-180375.csv")  # 原始SAR数据路径
        self.slice_csv_name = "sar_slices.csv"  # 切片数据文件名
        self.fit_csv_name = "sar_fit_degree.csv"  # 贴合度数据文件名
        self.md5_store_path = get_abs_path(chroma_conf["md5_hex_store"])  # MD5存储路径# 轻量级文本嵌入模型

        self.slice_collection = Chroma(
            collection_name=chroma_conf["collections"]["slices"],
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space": "cosine"}
        )

        
    def get_retriever(self, chromadb: Chroma):
        """
        获取向量检索器
        用于根据查询在向量数据库中搜索相关文档
        """
        return chromadb.as_retriever(search_kwargs={"k": chroma_conf["k"]})  # 返回前k个最相关的文档

    def check_md5_hex(self, md5_for_check: str) -> bool:
        """
        检查MD5值是否匹配
        参数:
            md5_for_check: 待验证的MD5值
        返回:
            是否匹配
        """
        if not md5_for_check: return False
        if not get_abs_path(self.md5_store_path): return False
        
        try:
            with open(self.md5_store_path, "r", encoding="utf-8") as f:
                return f.read().strip() == md5_for_check
        except Exception:
            return False

    def save_md5_hex(self, md5_for_check: str) -> None:
        """保存MD5值到文件"""
        if not md5_for_check: return
        
        with open(self.md5_store_path, "a", encoding="utf-8") as f:
            f.write(md5_for_check + "\n")

    def ensure_directories(self) -> None:
        """确保所有必要目录存在"""
        # 先判断目录是否存在
        if not os.path.exists(self.persist_directory):
            # 不存在则创建
            os.makedirs(self.persist_directory, exist_ok=True)
            # 输出你要的日志
            logger.info(f"[加载知识库]{self.persist_directory}目录不存在，已创建。")

    def build_sar_slice(self, csv_path: str = None, output_dir: str = None):
        """
        构建SAR切片和贴合度表
        参数:
            csv_path: 输入CSV路径，默认使用类初始化路径
            output_dir: 输出目录，默认使用类初始化目录
        返回:
            切片DataFrame, 贴合度DataFrame
        """
        self.ensure_directories()
        csv_path = csv_path or self.sar_csv_path
        output_dir = output_dir or self.data_dir

        # 检查数据文件是否存在
        if not os.path.exists(csv_path):
            logger.info(f"[加载知识库] SAR 数据文件不存在: {csv_path}。")
            raise FileNotFoundError(f"SAR 数据文件不存在: {csv_path}")

        # 计算文件MD5用于变更检测
        slice_out_path = os.path.join(output_dir, self.slice_csv_name)
        csv_md5 = get_file_md5_hex(csv_path)
        slice_md5 = get_file_md5_hex(slice_out_path)

        # 增量处理检查
        if (csv_md5 and slice_md5 
                and os.path.exists(slice_out_path) 
                and self.check_md5_hex(csv_md5) and self.check_md5_hex(slice_md5) 
                     ):
            logger.info(f"[加载知识库] SAR CSV 未发生变化，已存在处理结果，跳过重构。")
            slice_df = pd.read_csv(slice_out_path, encoding="utf-8-sig")
            return slice_df

        # 数据加载与预处理
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if "question" not in df.columns or "answer" not in df.columns:
            raise ValueError("SAR CSV 必须包含 'question' 和 'answer' 列")

        df["answer"] = df["answer"].fillna("")
        df["text"] = df["question"].astype(str) + " " + df["answer"].astype(str)  # 构建完整文本

        # 嵌入生成
        logger.info(f"[加载知识库] 正在生成文本语义嵌入...")
        embeddings = self.embeddings.embed_documents(df["text"].tolist())
        question_embeddings = self.embeddings.embed_documents(df["question"].tolist())

        # 聚类参数配置
        n_clusters = int(chroma_conf["clustering"]["n_clusters"])
        slice_size = int(chroma_conf["clustering"]["slice_size"])

        # 动态计算聚类大小范围
        n_samples = len(df)
        if n_clusters <= 0: n_clusters = 1
        
        base_size = max(1, n_samples // n_clusters)
        remainder = n_samples % n_clusters
        size_min = base_size
        size_max = base_size + 1 if remainder else base_size
        size_max = max(size_max, size_min)

        logger.info(f"[加载知识库] 目标簇数: {n_clusters}, 簇大小范围: {size_min}~{size_max}")
        logger.info(f"[加载知识库] 正在执行 KMeansConstrained 聚类...")

        # 执行带约束的聚类
        clf = KMeansConstrained(
            n_clusters=n_clusters,
            size_min=size_min,
            size_max=size_max,
            random_state=42,
            n_init=10,
            n_jobs=1,
        )
        df["cluster_id"] = clf.fit_predict(embeddings)

        # 生成切片和贴合度记录
        slice_records = []

        for cid in range(n_clusters):
            cluster_df = df[df["cluster_id"] == cid]
            if cluster_df.empty: continue

            cluster_indices = cluster_df.index.tolist()
            # 按切片大小分割簇
            for start in range(0, len(cluster_indices), slice_size):
                slice_idx = cluster_indices[start:start + slice_size]
                slice_data = cluster_df.loc[slice_idx]
                
                # 构建切片内容（Q&A对）
                slice_content = "\n".join([
                    f"Q: {row['question']} A: {row['answer']}"
                    for _, row in slice_data.iterrows()
                ])
                
                slice_id = f"sar_cluster_{cid}_slice_{start // slice_size}"
                slice_records.append({
                    "slice_id": slice_id,
                    "cluster_id": cid,
                    "slice_content": slice_content,
                })


        # 保存结果
        slice_df = pd.DataFrame(slice_records)
        slice_df.to_csv(slice_out_path, index=False, encoding="utf-8-sig")
        logger.info(f"[加载知识库] 已生成切片表: {slice_out_path}，共 {len(slice_df)} 条")

        # 保存MD5用于下次增量判断
        if csv_md5: self.save_md5_hex(csv_md5)

        return slice_df

    def build_chroma_collections(self, csv_path: str = None, output_dir: str = None):
        """构建Chroma向量数据库集合（修复SQL变量数超限问题）"""
        self.ensure_directories()
        csv_path = csv_path or self.sar_csv_path
        output_dir = output_dir or self.data_dir

        # 检查数据文件是否存在
        if not os.path.exists(csv_path):
            logger.info(f"[加载知识库]SAR 数据文件不存在: {csv_path}。")
            raise FileNotFoundError(f"SAR 数据文件不存在: {csv_path}")

        # 计算文件MD5用于变更检测
        slice_out_path = os.path.join(output_dir, self.slice_csv_name)
        csv_md5 = get_file_md5_hex(csv_path)
        slice_md5 = get_file_md5_hex(slice_out_path)

        # 增量处理检查
        if (csv_md5 and slice_md5 
                and os.path.exists(slice_out_path) 
                and self.check_md5_hex(csv_md5) and self.check_md5_hex(slice_md5) 
                     ):
            logger.info(f"[加载知识库]SAR CSV 未发生变化，已存在处理结果，跳过重构。")
            slice_df = pd.read_csv(slice_out_path, encoding="utf-8-sig")
            return slice_df

        # 加载切片和贴合度数据
        slice_file = os.path.join(self.data_dir, self.slice_csv_name)

        if not os.path.exists(slice_file):
            raise FileNotFoundError(f"切片表不存在: {slice_file}")

        slice_df = pd.read_csv(slice_file, encoding="utf-8-sig")

        logger.info(f"[加载知识库] 正在初始化 Chroma Embeddings...")

        # 初始化Chroma集合
        slice_collection = self.slice_collection

        def get_existing_ids_in_batches(collection, batch_size=500):
            """分批获取集合中的所有ID，规避SQL变量限制"""
            existing_ids = set()
            offset = 0
            while True:
                # 分批查询：每次取batch_size条
                batch_result = collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=[]  # 只获取ID，不获取文本/元数据，提升效率
                )
                batch_ids = batch_result.get('ids', [])
                if not batch_ids:
                    break
                existing_ids.update(batch_ids)
                offset += batch_size
            return existing_ids

        # 分批获取切片集合现有ID
        existing_slice_ids = get_existing_ids_in_batches(slice_collection)
        logger.info(f"[加载知识库] 基础切片库已有 {len(existing_slice_ids)} 条记录")
        
        new_slices = slice_df[~slice_df['slice_id'].isin(existing_slice_ids)]
        if not new_slices.empty:
            logger.info(f"[加载知识库] 需增量写入基础切片 {len(new_slices)} 条记录")
            
            # 分批处理提高大文件处理效率
            slice_texts = new_slices['slice_content'].tolist()
            slice_metadatas = [
                {
                    "cluster_id": int(row['cluster_id']),
                    "slice_id": str(row['slice_id'])
                } 
                for _, row in new_slices.iterrows()
            ]
            slice_ids = new_slices['slice_id'].tolist()
            
            batch_size = 500
            for i in tqdm(range(0, len(slice_texts), batch_size), desc="Writing Slices"):
                slice_collection.add_texts(
                    texts=slice_texts[i:i + batch_size],
                    metadatas=slice_metadatas[i:i + batch_size],
                    ids=slice_ids[i:i + batch_size],
                )

            logger.info(f"[加载知识库] 基础切片库写入完成！")
        else:
            logger.info(f"[加载知识库] 无需增量写入基础切片库")
        
        # 保存MD5用于下次增量判断
        slice_md5 = get_file_md5_hex(slice_out_path)
        if slice_md5: self.save_md5_hex(slice_md5)

        return slice_collection

    def build_all(self) -> None:
        """执行完整构建流程"""
        # self.build_sar_slice()
        self.build_chroma_collections()

        # 外部加在独立的MD5检查逻辑，确保每个阶段的增量处理都能正确记录和判断
        slice_out_path = os.path.join(self.data_dir, self.slice_csv_name)
        print(f"slice_out_path: {slice_out_path}")
        slice_md5 = get_file_md5_hex(slice_out_path)
        print(f"slice_md5: {slice_md5}")
        if slice_md5: self.save_md5_hex(slice_md5)

if __name__ == "__main__":
    # 命令行直接运行时的入口点
    builder = RscsvBuilder()
    builder.build_all()

    # 获取检索器
    retriever = builder.get_retriever(builder.slice_collection)

    # 测试查询：搜索包含"北京 弱电"的文档
    res = retriever.invoke("遥感 船只")
    
    # 打印检索结果
    for r in res:
        print(r.page_content)  # 文档内容
        print("-" * 20)  # 分隔线
