"""全局测试配置：agent/ 加入 sys.path；提供共享 fake 类与脚本加载器"""
import importlib.util
import sys
from pathlib import Path

# Windows 环境规避：模型栈（onnxruntime/torch）DLL 先加载会使 ortools 的
# WinDLL 加载失败（WinError 127，entry point not found）。rag 包导入链
# （slice_builder → k_means_constrained → ortools）必然触发，故在此预先加载。
import ortools  # noqa: E402,F401
import k_means_constrained  # noqa: E402,F401

_ROOT = Path(__file__).resolve().parents[1]
_AGENT = _ROOT / "agent"
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class NoopCollection:
    """不落盘的 Chroma fake：记录 add/delete，get 返回空"""
    def __init__(self):
        self.calls = []
        self.deleted = []

    def add_texts(self, texts=None, metadatas=None, ids=None):
        self.calls.append({"texts": texts, "metadatas": metadatas, "ids": ids})

    def delete(self, ids=None):
        self.deleted.extend(ids)

    def get(self, **kwargs):
        return {"ids": [], "documents": [], "metadatas": []}


class FakeSliceStore:
    """切片库 fake：get_by_ids 按 id 生成假内容"""
    def get_by_ids(self, ids):
        return {
            "ids": ids,
            "documents": [f"content_of_{i}" for i in ids],
            "metadatas": [{} for _ in ids],
        }


def load_script(name: str):
    """用 importlib 加载 agent/data/ 下的脚本（不依赖 agent.data 包结构）"""
    path = _ROOT / "agent" / "data" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
