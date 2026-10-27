"""服务卫生性回归测试：TTL 抖动与 request_id 链路标识。

与 test_service.py 不同，这里不依赖已训练的模型产物：
`TestClient(app)` 不进入 with 上下文时不会触发 lifespan，`/` 路由也不读取模型。
"""

from __future__ import annotations

from fastapi.testclient import TestClient

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
