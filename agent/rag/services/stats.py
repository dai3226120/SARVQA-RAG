"""
命中率统计追踪器
替代原 RscsvService 的类级全局变量，改为实例注入，更显式可测
"""
from dataclasses import dataclass


@dataclass
class HitRateTracker:
    """
    检索命中率统计器
    通过依赖注入由 MembershipHybridService 持有
    """

    _total_calls: int = 0
    _hit_calls: int = 0

    def record_call(self, is_hit: bool = False) -> None:
        """记录一次调用"""
        self._total_calls += 1
        if is_hit:
            self._hit_calls += 1

    @property
    def total_calls(self) -> int:
        return self._total_calls

    @property
    def hit_calls(self) -> int:
        return self._hit_calls

    @property
    def hit_rate(self) -> float:
        if self._total_calls == 0:
            return 0.0
        return self._hit_calls / self._total_calls

    def get_stats(self) -> dict:
        """获取统计数据"""
        return {
            "total_calls": self._total_calls,
            "hit_calls": self._hit_calls,
            "hit_rate": self.hit_rate,
        }

    def reset(self) -> None:
        """重置统计数据"""
        self._total_calls = 0
        self._hit_calls = 0
