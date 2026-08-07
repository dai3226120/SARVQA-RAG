# tests/test_metrics.py
"""评估指标回归测试

背景：calculate_cosine_similarity 原为 TF-IDF 词法重叠（同义改写即归零），
已改为嵌入向量语义相似度（实测均值 0.11 → 0.71，与 correct 相关性 0.245 → 0.293）。
"""
import pytest

from benchmark.core.metrics import calculate_cosine_similarity


def test_semantic_cosine_paraphrase_high():
    """同义改写应得到高相似度（旧 TF-IDF 词法实现会很低）"""
    s1 = "The cat sat on the mat."
    s2 = "The feline rested upon the rug."
    assert calculate_cosine_similarity(s1, s2) > 0.5


def test_semantic_cosine_exact_match_high():
    s = "A ship is visible in the harbor."
    assert calculate_cosine_similarity(s, s) > 0.9


def test_semantic_cosine_unrelated_low():
    s1 = "The cat sat on the mat."
    s2 = "The stock market rose sharply today."
    assert calculate_cosine_similarity(s1, s2) < 0.4


def test_semantic_cosine_empty_text_zero():
    assert calculate_cosine_similarity("", "some text") == 0.0
    assert calculate_cosine_similarity("some text", "") == 0.0
    assert calculate_cosine_similarity("", "") == 0.0
