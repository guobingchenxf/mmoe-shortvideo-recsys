"""延时统计工具。

在线服务对延时极其敏感（推荐接口通常要求 P99 < 100ms），
因此除了返回单次耗时，还需要在内存里维护一个滑动窗口，随时给出 P50/P95/P99，
便于压测和线上监控。
"""

from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager
from typing import Deque, Dict, Iterator, List, Optional


class LatencyTracker:
    """定长滑动窗口的延时统计器（线程不安全，单进程服务够用）。"""

    def __init__(self, window: int = 2000):
        self._samples: Deque[float] = deque(maxlen=window)
        self._total = 0

    def record(self, latency_ms: float) -> None:
        self._samples.append(float(latency_ms))
        self._total += 1

    @property
    def count(self) -> int:
        return self._total

    def stats(self) -> Dict[str, Optional[float]]:
        """返回 P50 / P90 / P95 / P99 / 均值 / 最大值（毫秒）。"""
        if not self._samples:
            return {"count": 0, "p50": None, "p90": None, "p95": None, "p99": None,
                    "mean": None, "max": None}
        arr = sorted(self._samples)
        n = len(arr)

        def pct(q: float) -> float:
            # 最近秩法：保证返回的是真实观测值，不做插值
            idx = min(n - 1, max(0, int(round(q * (n - 1)))))
            return arr[idx]

        return {
            "count": self._total,
            "window": n,
            "p50": round(pct(0.50), 3),
            "p90": round(pct(0.90), 3),
            "p95": round(pct(0.95), 3),
            "p99": round(pct(0.99), 3),
            "mean": round(sum(arr) / n, 3),
            "max": round(arr[-1], 3),
        }

    def reset(self) -> None:
        self._samples.clear()
        self._total = 0


class Timer:
    """请求级计时器：可记录多个阶段的分段耗时。"""

    def __init__(self):
        self._t0 = time.perf_counter()
        self.marks: List[Dict[str, float]] = []

    def mark(self, name: str) -> float:
        """记录一个阶段，返回该阶段耗时（毫秒）。"""
        now = time.perf_counter()
        last = self.marks[-1]["_t"] if self.marks else self._t0
        cost = (now - last) * 1000.0
        self.marks.append({"name": name, "cost_ms": round(cost, 3), "_t": now})
        return cost

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def breakdown(self) -> Dict[str, float]:
        return {m["name"]: m["cost_ms"] for m in self.marks}


@contextmanager
def timeit() -> Iterator[Dict[str, float]]:
    """上下文管理器：with timeit() as t: ... ; t['ms'] 即耗时（毫秒）。"""
    box: Dict[str, float] = {}
    t0 = time.perf_counter()
    try:
        yield box
    finally:
        box["ms"] = (time.perf_counter() - t0) * 1000.0
