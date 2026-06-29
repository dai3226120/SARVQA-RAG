from utils.file_handler import md_loader

docs = md_loader(r"C:\dataset\SARVQA-RAG\agent\data\remote_sensing_wiki.md")
print("总文档数:", len(docs))
if docs:
    print("第一篇内容样例:\n", docs[0].page_content[:5000]) # 打印前500字看看有没有正文