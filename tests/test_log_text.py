from rag.membership.log_manager import build_log_doc_text, fetch_slices_content
from tests.conftest import FakeSliceStore


def test_three_sections():
    t = build_log_doc_text("q?", ["slice one"], "answer text")
    assert t == "Question: q?\nRetrieved Slices: slice one\nAnswer: answer text"


def test_multiple_slices_join_with_newline():
    t = build_log_doc_text("q?", ["s1", "s2"], "a")
    assert "Retrieved Slices: s1\ns2\nAnswer: a" in t


def test_per_slice_truncation():
    t = build_log_doc_text("q?", ["x" * 300], "a", slice_char_limit=200)
    # 每片截 200 字符
    assert "x" * 200 in t
    assert "x" * 201 not in t


def test_total_truncation():
    t = build_log_doc_text("q?" + "y" * 300, ["s"], "a", total_char_limit=500)
    assert len(t) <= 500


def test_empty_answer():
    t = build_log_doc_text("q?", ["s"], "")
    assert t.endswith("Answer: ")


def test_fetch_slices_content_joins():
    store = FakeSliceStore()
    assert fetch_slices_content(store, ["s1", "s2"]) == "content_of_s1\ncontent_of_s2"


def test_fetch_slices_content_empty():
    store = FakeSliceStore()
    assert fetch_slices_content(store, []) == ""
    assert fetch_slices_content(store, ["", " "]) == ""
