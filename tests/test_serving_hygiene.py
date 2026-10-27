"""服务卫生性回归测试：TTL 抖动、request_id 链路标识、双缓冲热加载。

与 test_service.py 不同，这里不依赖已训练的模型产物：
`TestClient(app)` 不进入 with 上下文时不会触发 lifespan，`/` 路由也不读取模型；
热加载部分用假实例验证，不需要真实的 Recommender。
"""

from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

from recsys.services.hot_reload import RecommenderHolder, ReloadInProgress
from recsys.services.main import app
from recsys.services.ranking import _jittered_ttl


# ---------------- TTL 抖动 ----------------


def test_jittered_ttl_stays_in_range() -> None:
    """抖动幅度应限制在 ±10% 附近（允许浮点取整的 1 个单位偏差）。"""
    values = [_jittered_ttl(3600) for _ in range(200)]
    assert all(3239 <= v <= 3961 for v in values)


def test_jittered_ttl_actually_varies() -> None:
    """关键行为：同一批写入的 key 不能拿到完全相同的 TTL，否则抖动等于没做。"""
    values = {_jittered_ttl(3600) for _ in range(200)}
    assert len(values) > 5


def test_jittered_ttl_non_positive_returns_zero() -> None:
    """非正 TTL 返回 0，交由调用方沿用默认 TTL。"""
    assert _jittered_ttl(0) == 0
    assert _jittered_ttl(-1) == 0


# ---------------- request_id 中间件 ----------------


def _client() -> TestClient:
    """构造 TestClient 但**不进入 with 上下文**。

    with 会执行 lifespan（加载模型与物料），而这里只想验证中间件；
    直接发请求即可，`/` 路由本身不依赖模型。
    """
    return TestClient(app, raise_server_exceptions=False)


def test_access_log_generates_request_id() -> None:
    """未传 X-Request-ID 时，中间件应生成一个 12 位十六进制标识。"""
    client = _client()
    try:
        resp = client.get("/")
    finally:
        client.close()

    assert resp.status_code == 200
    assert len(resp.headers["X-Request-ID"]) == 12
    assert float(resp.headers["X-Cost-Ms"]) >= 0.0


def test_access_log_propagates_incoming_request_id() -> None:
    """上游已带 X-Request-ID 时应原样透传，保证跨服务链路可串。"""
    client = _client()
    try:
        resp = client.get("/", headers={"X-Request-ID": "trace-abc"})
    finally:
        client.close()

    assert resp.headers["X-Request-ID"] == "trace-abc"


# ---------------- 双缓冲热加载 ----------------


class _FakeRecommender:
    """只实现 holder 需要的两个行为：close，以及可比较的身份。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_reload_swaps_instance_and_closes_idle_old() -> None:
    old = _FakeRecommender("old")
    new = _FakeRecommender("new")
    holder = RecommenderHolder(old, lambda: new)

    result = holder.reload(timeout=0.5)

    assert holder.current is new
    assert holder.generation == 1
    assert result["drained"] is True
    assert result["old_closed"] is True
    assert old.closed is True


def test_reload_keeps_serving_when_build_fails() -> None:
    """构建失败必须保住旧实例——这是热加载最重要的安全性质。"""
    old = _FakeRecommender("old")

    def boom() -> _FakeRecommender:
        raise RuntimeError("build failed")

    holder = RecommenderHolder(old, boom)

    with pytest.raises(RuntimeError):
        holder.reload()

    assert holder.current is old
    assert holder.generation == 0
    assert old.closed is False


def test_reload_defers_close_while_request_in_flight() -> None:
    """在途请求未归还时不能关旧实例，否则会掐断正在使用的连接。"""
    old = _FakeRecommender("old")
    new = _FakeRecommender("new")
    holder = RecommenderHolder(old, lambda: new, drain_timeout=0.05)

    entered = threading.Event()
    release = threading.Event()

    def worker() -> None:
        with holder.acquire():
            entered.set()
            release.wait(5.0)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    assert entered.wait(5.0), "工作线程未能进入临界区"

    result = holder.reload()

    assert result["drained"] is False
    assert old.closed is False, "有在途请求时不应释放旧实例"

    release.set()
    thread.join(5.0)
    assert old.closed is True, "最后一个请求归还后应由它释放旧实例"


def test_reload_rejects_concurrent_calls() -> None:
    """单飞锁：并发 reload 会把内存叠成多份，必须拒绝。"""
    old = _FakeRecommender("old")
    started = threading.Event()
    release = threading.Event()

    def slow_factory() -> _FakeRecommender:
        started.set()
        release.wait(5.0)
        return _FakeRecommender("new")

    holder = RecommenderHolder(old, slow_factory)
    thread = threading.Thread(target=holder.reload, daemon=True)
    thread.start()
    assert started.wait(5.0), "慢构建未开始"

    with pytest.raises(ReloadInProgress):
        holder.reload()

    release.set()
    thread.join(5.0)
