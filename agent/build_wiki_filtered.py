"""
wiki 语料领域筛选脚本

问题：wiki 按主题抓取的主题漂移严重（数学谱方法、地缘诗学、脑图谱等混入），
导致检索注入无关上下文。本脚本：
1. MarkdownHeaderTextSplitter 按标题层级切分（保留 meta 标题）
2. 章节筛选：标题命中遥感正关键词（SAR/radar/遥感/卫星/图像判读等）才保留
3. 内容筛选：内容中遥感关键词命中数 >= 阈值，剔除碎片与低相关段落
4. 重组筛选后的 md（保持标题层级）输出到 agent/data/wiki_filtered/

输出供 KnowledgeBuilder 重建入库。
"""
import os
import re
import sys

import k_means_constrained  # noqa: F401

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
for p in (current_dir, root_dir):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from langchain_text_splitters import MarkdownHeaderTextSplitter

ARCHIVE_DIR = os.path.join(current_dir, "data", "archive_wiki")
OUT_DIR = os.path.join(current_dir, "data", "wiki_filtered")

# ---- 正关键词（遥感/SAR 领域）----
POSITIVE = [
    "sar", "synthetic aperture radar", "remote sensing", "radar", "backscatter",
    "satellite", "imagery", "image interpretation", "earth observation",
    "polarimetric", "insar", "interferometry", "land cover", "land use",
    "crop monitoring", "vegetation index", "water body", "coastal", "ocean",
    "urban", "building", "agriculture", "forest", "flood", "disaster monitoring",
    "hyperspectral", "multispectral", "thermal imaging", "microwave",
    "scene classification", "object detection", "change detection",
    "sensor", "resolution", "pixel", "aerial photography", "photogrammetry",
    "soil moisture", "topography", "landform", "geomorphology", "sea state",
    "ship detection", "mapping", "monitoring", "spectral signature", "reflectance",
]
# ---- 负关键词（标题含任一即剔除整节）----
NEGATIVE_TITLE = [
    "spectral method", "differential equation", "numerical analysis",
    "brain", "neuro", "geopoetic", "poetry", "literature", "music",
    "sport", "film", "video game", "food", "religion", "philosophy",
    "politics", "economy", "medicine", "disease", "anatomy", "animal",
    "bird", "insect", "history of", "language", "linguistic", "culture",
]

# 标题层级切分（与 knowledge_builder 一致）
HEADERS_TO_SPLIT = [
    ("#", "Knowledge_Base"),
    ("##", "Main_Topic"),
    ("###", "Sub_Section"),
    ("####", "Sub_Sub_Section"),
    ("#####", "Detail_Section"),
]


def _title_keywords(section) -> str:
    """汇总章节的全部标题层级文本（含 Knowledge_Base 根标题）"""
    parts = []
    for key in ("Knowledge_Base", "Main_Topic", "Sub_Section", "Sub_Sub_Section", "Detail_Section"):
        v = section.metadata.get(key)
        if v:
            parts.append(str(v))
    return " | ".join(parts)


def _score(text: str) -> int:
    """遥感关键词命中计数（去重词）"""
    t = text.lower()
    hits = set()
    for kw in POSITIVE:
        if kw in t:
            hits.add(kw)
    return len(hits)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT, strip_headers=False)

    total_in = 0
    total_kept = 0
    for fname in sorted(os.listdir(ARCHIVE_DIR)):
        if not fname.endswith(".md"):
            continue
        path = os.path.join(ARCHIVE_DIR, fname)
        text = open(path, encoding="utf-8").read()
        sections = splitter.split_text(text)
        total_in += len(sections)

        kept_parts = []
        for sec in sections:
            title = _title_keywords(sec)
            content = sec.page_content

            # 1) 标题负关键词剔除
            tl = title.lower()
            if any(neg in tl for neg in NEGATIVE_TITLE):
                continue
            # 2) 标题正关键词：标题命中即保留候选（SAR/遥感直接主题）
            title_score = _score(title)
            content_score = _score(content)
            # 3) 内容过短碎片剔除
            if len(content.strip()) < 100:
                continue
            # 保留规则：标题命中 >=1 且 内容命中 >=1；或标题命中 >=2
            if title_score >= 2 or (title_score >= 1 and content_score >= 1):
                kept_parts.append(f"{title}\n\n{content.strip()}\n")
                total_kept += 1

        # 重组输出（保留原标题层级）
        out_path = os.path.join(OUT_DIR, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(kept_parts))
        print(f"{fname}: {len(sections)} 节 -> 保留 {len(kept_parts)} 节 ({os.path.getsize(out_path)//1024} KB)")

    print(f"\n合计: {total_in} 节 -> 保留 {total_kept} 节")


if __name__ == "__main__":
    main()
