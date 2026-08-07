# tests/test_llm_judge.py
import pytest

from rag.membership.llm_judge import LlmJudge


class FakeModel:
    """模拟 LLM：前 fail_times 次调用抛异常，之后返回固定结果"""

    def __init__(self, fail_times=0, result="1"):
        self.fail_times = fail_times
        self.result = result
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(
                "Error code: 500 - {'error': {'code': 'InternalServiceError'}}"
            )
        return type("Response", (), {"content": self.result})()


def test_judge_succeeds_without_retry():
    model = FakeModel(fail_times=0, result="1")
    judge = LlmJudge(model=model, max_retries=3, retry_delay=0.01)
    assert judge.judge("q?", "ground truth", "predicted") == "1"
    assert model.calls == 1


def test_judge_retries_then_succeeds():
    """前 2 次 500 失败，第 3 次成功 → 返回 '1' 且调用 3 次"""
    model = FakeModel(fail_times=2, result="1")
    judge = LlmJudge(model=model, max_retries=3, retry_delay=0.01)
    assert judge.judge("q?", "ground truth", "predicted") == "1"
    assert model.calls == 3


def test_judge_returns_zero_after_all_retries_fail():
    """全部重试失败 → 返回 '0' 且恰好调用 max_retries 次"""
    model = FakeModel(fail_times=99)
    judge = LlmJudge(model=model, max_retries=3, retry_delay=0.01)
    assert judge.judge("q?", "ground truth", "predicted") == "0"
    assert model.calls == 3


def test_judge_zero_result_parsed():
    model = FakeModel(fail_times=0, result="0")
    judge = LlmJudge(model=model, max_retries=3, retry_delay=0.01)
    assert judge.judge("q?", "ground truth", "predicted") == "0"
    assert model.calls == 1
