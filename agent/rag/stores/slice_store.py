"""
SAR 切片向量存储
实现 BaseVectorStore，对应原 RscsvBuilder.slice_collection
"""
from typing import Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.base.vector_store import BaseVectorStore
from rag.core.config import rag_config
from model.factory import huggingface_embed_model


class SliceStore(BaseVectorStore):
    """SAR 问答切片向量存储"""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        embedding_function=None,
    ):
        self._collection = Chroma(
            collection_name=collection_name or rag_config.slices_collection_name,
            embedding_function=embedding_function or huggingface_embed_model,
            persist_directory=persist_directory or rag_config.persist_directory,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def get_retriever(self, k: int = 5):
        return self._collection.as_retriever(search_kwargs={"k": k})

    def add_documents(self, documents: list[Document], batch_size: int = 500) -> int:
        """分批写入文档"""
        count = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            self._collection.add_documents(batch)
            count += len(batch)
        return count

    def get_by_ids(self, ids: list[str]) -> dict:
        return self._collection.get(ids=ids)

    def count(self) -> int:
        total = 0
        offset = 0
        while True:
            result = self._collection.get(limit=500, offset=offset, include=[])
            ids = result.get("ids", [])
            if not ids:
                break
            total += len(ids)
            offset += 500
        return total

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        return self._collection.similarity_search(query, k=k)

    def similarity_search_with_scores(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        return self._collection.similarity_search_with_relevance_scores(query, k=k)

    @property
    def collection(self):
        return self._collection
