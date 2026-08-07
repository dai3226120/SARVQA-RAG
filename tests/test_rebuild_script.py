# tests/test_rebuild_script.py
"""向量库重建脚本冒烟测试：脚本可加载、三个重建函数与 CLI 入口存在"""
import pytest

from tests.conftest import load_script


def test_rebuild_script_loads_and_exports():
    mod = load_script("rebuild_vector_stores")
    assert callable(mod.rebuild_slices)
    assert callable(mod.rebuild_knowledge)
    assert callable(mod.rebuild_logs)
    assert callable(mod.main)
    assert set(mod._BUILDERS.keys()) == {"slices", "knowledge", "logs"}
