from langchain_chroma import Chroma
from langchain_community.llms.tongyi import Tongyi
from sqlalchemy.orm.collections import collection
from utils.config_handler import chroma_conf
from model.factory import huggingface_embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter  
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, md_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
from langchain_core.documents import Document
import os
import time
import pandas as pd  # 💡 引入 pandas 用于导出 CSV

# 全局禁用SSL警告，解决通义API网络问题
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=huggingface_embed_model,
            persist_directory=get_abs_path(chroma_conf["persist_diretory"])
        )

        # 1. 通用文本细粒度切分器（当 Markdown 章节过长时进行二次平滑切分）
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

        # 2. 💡【核心修改】完美适配全新的 RAG 维基语料库标准层级
        # 将 Markdown 标题精准映射为语义化的 Metadata 标签
        headers_to_split_on = [
            ("#", "Knowledge_Base"),       # 顶层库名 (Remote Sensing Knowledge Base)
            ("##", "Main_Topic"),          # 对应维基主词条名 (如: Remote sensing)
            ("###", "Sub_Section"),        # 对应二级章节 (如: Overview, History)
            ("####", "Sub_Sub_Section"),   # 对应三级更细章节
            ("#####", "Detail_Section")     # 对应叶子节点微观章节
        ]
        
        self.md_header_spliter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False  # 保留文本内的标题符号，有助于大模型理解上下文结构
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def load_document(self, export_csv_path: str = "knowledge_chunks.csv"):
        """
        加载文档入库，并可选将本次解析的所有分块导出为 CSV 格式
        :param export_csv_path: 导出的 CSV 文件路径
        """
        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    if line.strip() == md5_for_check:
                        return True
            return False

        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)
            if read_path.endswith("pdf"):
                return pdf_loader(read_path)
            if read_path.endswith("md"):
                return md_loader(read_path)
            return []

        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        BATCH_SIZE = 500  
        all_processed_chunks = []  # 用于收集所有分块数据以便导出 CSV

        for path in allowed_files_path:
            md5_hex = get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}已存在，跳过")
                continue

            try:
                documents = get_file_documents(path)
                if not documents:
                    logger.warning(f"[加载知识库]{path}无有效内容")
                    continue

                if path.endswith("md"):
                    split_documents = []
                    for doc in documents:
                        # 💡 步骤A：先按维基专属标题层级切分
                        md_header_splits = self.md_header_spliter.split_text(doc.page_content)
                        # 💡 步骤B：对过长章节启动 Recursive 二次切分（防止单块 Token 溢出，同时继承大标题元数据）
                        md_recursive_splits = self.spliter.split_documents(md_header_splits)
                        
                        for sub_doc in md_recursive_splits:
                            sub_doc.metadata.update(doc.metadata)
                        
                        split_documents.extend(md_recursive_splits)
                else:
                    split_documents = self.spliter.split_documents(documents)

                if not split_documents:
                    logger.warning(f"[加载知识库]{path}无有效内容")
                    continue

                # 收集当前文件的分块信息
                all_processed_chunks.extend(split_documents)

                total = len(split_documents)
                logger.info(f"[加载知识库]{path} 分块完成，共 {total} 块，开始分批入库")

                # 自动分批入库
                for i in range(0, total, BATCH_SIZE):
                    batch = split_documents[i:i + BATCH_SIZE]
                    self.vector_store.add_documents(batch)
                    logger.info(f"[分批入库] 已完成 {min(i+BATCH_SIZE, total)}/{total} 块")
                    time.sleep(0.1)  

                save_md5_hex(md5_hex)
                logger.info(f"[加载知识库]{path} 全部加载成功 ✅")

            except Exception as e:
                logger.error(f"[加载知识库]{path}加载失败: {str(e)}", exc_info=True)
                continue

        # 将收集到的分块数据转换并导出为 CSV
        if all_processed_chunks:
            self._export_to_csv(all_processed_chunks, export_csv_path)

    def _export_to_csv(self, chunks: list[Document], csv_path: str):
        """ 内部辅助函数：将 Document 列表平铺并使用 pandas 写入 CSV """
        try:
            csv_data = []
            for idx, doc in enumerate(chunks):
                row = {
                    "chunk_id": idx + 1,
                    "content": doc.page_content
                }
                # 动态展开元数据
                if doc.metadata:
                    for k, v in doc.metadata.items():
                        row[f"meta_{k}"] = v
                csv_data.append(row)
            
            df = pd.DataFrame(csv_data)
            
            abs_csv_path = get_abs_path(csv_path)
            df.to_csv(abs_csv_path, index=False, encoding="utf-8-sig")
            logger.info(f"[CSV导出成功] 所有的文档分块已保存至: {abs_csv_path} 📊")
        except Exception as e:
            logger.error(f"[CSV导出失败] {str(e)}", exc_info=True)


if __name__ == "__main__":
    vs = VectorStoreService()
    
    # 运行加载并自动导出 CSV (默认保存在当前根目录下的 knowledge_chunks.csv)
    vs.load_document(export_csv_path="knowledge_chunks.csv")

    print("\n--- 检索测试 ---")
    retriever = vs.get_retriever()
    res = retriever.invoke("sar图像中的飞机目标提取方法有哪些？")
    for r in res:
        print("【检索文本】:", r.page_content[:150], "...")
        print("【提取元数据】:", r.metadata)  
        print("-" * 30)