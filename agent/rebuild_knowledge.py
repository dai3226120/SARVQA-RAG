"""
临时脚本：重建知识库（知识库语料已从 wiki 换为 SAR 领域论文）

步骤：
1. 删除 Chroma 中旧的知识库 collection（仅 rag_config.collection_name，不动 slices）
2. 运行 KnowledgeBuilder 重新入库 agent/data 下的语料（当前为 pdf_text_v2.txt）
"""
import os
import sys

# Windows 下 pyarrow(pandas 间接依赖) 先加载会导致 ortools DLL 加载失败(WinError 127)。
# 必须先于 pandas 导入 k_means_constrained，让 ortools 扩展 DLL 先进入进程。
import k_means_constrained  # noqa: F401

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
for p in (current_dir, root_dir):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rag.core.config import rag_config
from rag.stores.knowledge_store import KnowledgeStore
from rag.builders.knowledge_builder import KnowledgeBuilder

COLLECTION = rag_config.collection_name


def main():
    # 1. 删除旧知识库 collection
    store = KnowledgeStore()
    print(f"删除前 knowledge collection 文档数: {store.count()}")
    client = store._collection._client
    client.delete_collection(COLLECTION)
    print(f"✅ 已删除 collection: {COLLECTION}")

    # 2. 重建（md5 已注册的 wiki 文件不在 data_path，不会再入库）
    builder = KnowledgeBuilder()
    builder.load_document(export_csv_path="agent/knowledge_chunks.csv")

    # 3. 统计
    new_store = KnowledgeStore()
    print(f"\n重建后 knowledge collection 文档数: {new_store.count()}")


if __name__ == "__main__":
    main()
