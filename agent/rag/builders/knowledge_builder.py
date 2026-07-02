"""
知识库构建器
负责文档加载、Markdown 标题层级切片、通用文本二次切分、向量库入库、CSV 导出
源自原 vector_store.py (VectorStoreService)
"""
import os
import time
import pandas as pd

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_core.documents import Document

from rag.stores.knowledge_store import KnowledgeStore
from rag.core.config import rag_config
from rag.core.md5_store import Md5Store
from utils.file_handler import (
    pdf_loader,
    txt_loader,
    md_loader,
    listdir_with_allowed_type,
    get_file_md5_hex,
)
from utils.logger_handler import logger


# 全局禁用 SSL 警告，解决通义 API 网络问题
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class KnowledgeBuilder:
    """通用知识库构建器，将文档文件切分后入库到 KnowledgeStore"""

    def __init__(self):
        self._store = KnowledgeStore()
        self._md5_store = Md5Store(rag_config.md5_store_path)

        # 通用文本细粒度切分器（当 Markdown 章节过长时进行二次平滑切分）
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=rag_config.chunk_size,
            chunk_overlap=rag_config.chunk_overlap,
            separators=rag_config.separators,
            length_function=len,
        )

        # Markdown 标题层级切分器（适配维基语料库标准层级）
        headers_to_split_on = [
            ("#", "Knowledge_Base"),
            ("##", "Main_Topic"),
            ("###", "Sub_Section"),
            ("####", "Sub_Sub_Section"),
            ("#####", "Detail_Section"),
        ]
        self._md_header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )

    @property
    def store(self) -> KnowledgeStore:
        return self._store

    def load_document(self, export_csv_path: str = "knowledge_chunks.csv"):
        """
        加载文档入库，并可选将本次解析的所有分块导出为 CSV

        Args:
            export_csv_path: 导出的 CSV 文件路径
        """
        allowed_files = listdir_with_allowed_type(
            rag_config.data_path,
            tuple(rag_config.allow_knowledge_file_type),
        )

        BATCH_SIZE = 500
        all_processed_chunks = []

        for path in allowed_files:
            md5_hex = get_file_md5_hex(path)
            if self._md5_store.is_registered(md5_hex):
                logger.info(f"[加载知识库]{path}已存在，跳过")
                continue

            try:
                documents = self._load_file(path)
                if not documents:
                    logger.warning(f"[加载知识库]{path}无有效内容")
                    continue

                split_documents = self._split(documents, path)
                if not split_documents:
                    logger.warning(f"[加载知识库]{path}无有效内容")
                    continue

                all_processed_chunks.extend(split_documents)
                total = len(split_documents)
                logger.info(f"[加载知识库]{path} 分块完成，共 {total} 块，开始分批入库")

                for i in range(0, total, BATCH_SIZE):
                    batch = split_documents[i : i + BATCH_SIZE]
                    self._store.add_documents(batch)
                    logger.info(f"[分批入库] 已完成 {min(i + BATCH_SIZE, total)}/{total} 块")
                    time.sleep(0.1)

                self._md5_store.register(md5_hex)
                logger.info(f"[加载知识库]{path} 全部加载成功 ✅")

            except Exception as e:
                logger.error(f"[加载知识库]{path}加载失败: {str(e)}", exc_info=True)
                continue

        if all_processed_chunks:
            self._export_to_csv(all_processed_chunks, export_csv_path)

    def _load_file(self, path: str) -> list[Document]:
        """根据文件类型加载文档"""
        if path.endswith("txt"):
            return txt_loader(path)
        if path.endswith("pdf"):
            return pdf_loader(path)
        if path.endswith("md"):
            return md_loader(path)
        return []

    def _split(self, documents: list[Document], path: str) -> list[Document]:
        """对文档进行切分"""
        if path.endswith("md"):
            split_documents = []
            for doc in documents:
                # 步骤A：按 Markdown 标题层级切分
                md_header_splits = self._md_header_splitter.split_text(doc.page_content)
                # 步骤B：对过长章节二次切分
                md_recursive_splits = self._splitter.split_documents(md_header_splits)
                for sub_doc in md_recursive_splits:
                    sub_doc.metadata.update(doc.metadata)
                split_documents.extend(md_recursive_splits)
        else:
            split_documents = self._splitter.split_documents(documents)
        return split_documents

    def _export_to_csv(self, chunks: list[Document], csv_path: str):
        """将 Document 列表导出为 CSV"""
        try:
            csv_data = []
            for idx, doc in enumerate(chunks):
                row = {"chunk_id": idx + 1, "content": doc.page_content}
                if doc.metadata:
                    for k, v in doc.metadata.items():
                        row[f"meta_{k}"] = v
                csv_data.append(row)

            df = pd.DataFrame(csv_data)
            from utils.path_tool import get_abs_path

            abs_csv_path = get_abs_path(csv_path)
            df.to_csv(abs_csv_path, index=False, encoding="utf-8-sig")
            logger.info(f"[CSV导出成功] 所有的文档分块已保存至: {abs_csv_path} 📊")
        except Exception as e:
            logger.error(f"[CSV导出失败] {str(e)}", exc_info=True)
