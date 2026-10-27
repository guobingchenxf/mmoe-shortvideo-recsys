"""双缓冲模型热加载：不中断在线服务地替换 Recommender 实例。

为什么需要双缓冲
----------------
直接 `app.state.recommender = new` 有两个问题：

1. 构建新模型要几十秒（加载权重、重建召回索引、灌缓存），这段时间请求没有服务可用；
2. 即便构建完成，旧实例上可能仍有在途请求在使用它的对象与 Redis 连接，
   立刻 `close()` 会把正在被使用的连接关掉。

双缓冲的做法：先在后台把新实例完整构建并预热好，再用**一次原子引用交换**切换；
旧实例标记为「退役」，等在途请求全部归还之后再释放。

代价：切换瞬间新旧两份同时驻留，线上要评估 2x 模型内存。
"""

from __future__ import annotations

import hmac
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional

from fastapi import FastAPI, Header, HTTPException

from recsys.utils.logger import get_logger

logger = get_logger("recsys.services.hot_reload")

#: 等旧实例排空的默认时长（秒）。超过则不再等，改由最后一个归还的请求负责释放。
DEFAULT_DRAIN_TIMEOUT = 5.0

#: 校验热加载接口的环境变量名。
TOKEN_ENV = "RELOAD_TOKEN"


class _Entry:
    """一个「代」的实例 + 它的在途引用计数。"""

    __slots__ = ("instance", "refs", "retired", "closed")

    def __init__(self, instance: Any) -> None:
        self.instance = instance
        self.refs = 0
        self.retired = False
        self.closed = False


def _safe_close(instance: Any) -> None:
    """释放旧实例。释放失败只记日志，不能影响已经生效的新实例。"""
    try:
        instance.close()
    except Exception:
        logger.exception("旧实例释放失败")


class ReloadInProgress(RuntimeError):
    """已有一次热加载在进行（用于和「构建失败」区分开，前者回 409，后者回 500）。"""


class RecommenderHolder:
    """持有当前的 Recommender，支持原子替换与在途请求引用计数。

    读路径（acquire）只在一把条件锁里做计数自增，不做任何 IO；
    写路径（reload）由单飞锁保证同一时刻只有一次加载，
    避免并发 reload 把内存叠成 3 份、4 份。
    """

    def __init__(
        self,
        current: Any,
        factory: Callable[[], Any],
        drain_timeout: float = DEFAULT_DRAIN_TIMEOUT,
    ) -> None:
        self._entry = _Entry(current)
        self._factory = factory
        self._drain_timeout = drain_timeout
        self._cond = threading.Condition(threading.Lock())
        self._reload_guard = threading.Lock()
        self._generation = 0

    @property
    def current(self) -> Any:
        """当前生效的实例（不加锁：属性读取是原子的，拿到旧值也不会出错）。"""
        return self._entry.instance

    @property
    def generation(self) -> int:
        return self._generation

    @contextmanager
    def acquire(self) -> Iterator[Any]:
        """借出当前实例，保证在 `with` 期间它不会被释放。

        请求进来时绑定到某一代实例上；即使中途发生热加载，这个请求
        也会继续用旧实例跑完，而不是在推理途中被换掉。
        """
        with self._cond:
            entry = self._entry
            entry.refs += 1
        try:
            yield entry.instance
        finally:
            should_close = False
            with self._cond:
                entry.refs -= 1
                if entry.refs == 0:
                    self._cond.notify_all()
                    # 退役实例的最后一个请求归还时，由它负责释放
                    if entry.retired and not entry.closed:
                        entry.closed = True
                        should_close = True
            if should_close:
                _safe_close(entry.instance)

    def reload(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """构建新实例并切换。构建失败时旧实例继续服务（异常原样抛出）。"""
        if not self._reload_guard.acquire(blocking=False):
            raise ReloadInProgress("已有一次热加载在进行，请稍后再试")
        try:
            t0 = time.perf_counter()
            new_instance = self._factory()  # 失败在这里抛，此时新旧都还没动
            load_ms = (time.perf_counter() - t0) * 1000.0

            new_entry = _Entry(new_instance)
            with self._cond:
                old_entry = self._entry
                old_entry.retired = True
                self._entry = new_entry
                self._generation += 1
                generation = self._generation

            drained = self._drain(old_entry, self._drain_timeout if timeout is None else timeout)

            old_closed = False
            if drained:
                with self._cond:
                    if not old_entry.closed:
                        old_entry.closed = True
                        old_closed = True
                if old_closed:
                    _safe_close(old_entry.instance)
            else:
                logger.warning(
                    "旧实例仍有 %d 个在途请求，将由最后一个归还的请求释放",
                    old_entry.refs,
                )

            logger.info(
                "热加载完成: generation=%d load=%.0fms drained=%s", generation, load_ms, drained
            )
            return {
                "generation": generation,
                "load_ms": round(load_ms, 1),
                "drained": drained,
                "old_closed": old_closed,
            }
        finally:
            self._reload_guard.release()

    def _drain(self, entry: _Entry, timeout: float) -> bool:
        """等到该代实例的在途请求归零；超时返回 False（不阻塞太久）。"""
        deadline = time.monotonic() + max(0.0, timeout)
        with self._cond:
            while entry.refs > 0 and not entry.closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
        return True

    def stats(self) -> Dict[str, Any]:
        with self._cond:
            entry = self._entry
            return {"generation": self._generation, "inflight": entry.refs}


def install_reload_routes(
    app: FastAPI, path: str = "/admin/reload", token_env: str = TOKEN_ENV
) -> None:
    """注册热加载接口。

    未配置 `RELOAD_TOKEN` 时接口直接返回 503：无鉴权的 reload 端点是一个
    现成的 DoS 入口——反复调用会把内存叠成多份。
    实例从 `app.state.holder` 惰性获取，所以本函数可以在 lifespan 之前调用。
    """

    @app.post(path, summary="热加载模型（双缓冲）")
    def reload_model(
        x_reload_token: Optional[str] = Header(default=None, alias="X-Reload-Token"),
    ) -> Dict[str, Any]:
        # 同步 def：加载要几十秒，FastAPI 会丢到线程池，不占事件循环。
        # 代价是加载期间会占用一个线程池名额，这也限制了并发 reload 的实际危害。
        expected = os.getenv(token_env)
        if not expected:
            raise HTTPException(status_code=503, detail=f"未配置 {token_env}，热加载接口已禁用")
        if not hmac.compare_digest((x_reload_token or "").encode(), expected.encode()):
            raise HTTPException(status_code=403, detail="X-Reload-Token 不正确")

        holder = getattr(app.state, "holder", None)
        if holder is None:
            raise HTTPException(status_code=503, detail="服务尚未就绪")

        try:
            return holder.reload()
        except ReloadInProgress as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("热加载失败，继续使用旧实例")
            raise HTTPException(status_code=500, detail=f"热加载失败: {exc}") from exc
