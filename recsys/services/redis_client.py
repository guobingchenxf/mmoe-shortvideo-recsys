"""用户行为序列缓存客户端：统一接口 + 内存降级。

=====================================================================
【为什么要有这一层抽象】
=====================================================================
线上推荐服务需要低延时读取用户"最近行为序列"（用于兴趣建模），
生产环境一般用 Redis。但本项目的目标是"没装 Redis 也能一键跑通全套流程"，
所以这里定义同一个 CacheClient 接口，提供两种实现：

    RedisCacheClient     设置 REDIS_URL 时启用（生产）
    InMemoryCacheClient  默认（本地开发 / 无 Redis 环境）

好处：业务代码只依赖抽象接口，切换后端不需要改一行业务逻辑——
这也是"依赖倒置"在工程里的典型用法。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from recsys.utils.logger import get_logger

logger = get_logger("recsys.services.cache")

#: 缓存 key 前缀，避免与同一 Redis 实例上的其他业务冲突
KEY_PREFIX = "recsys:user_seq:"


class CacheClient(ABC):
    """用户行为序列缓存抽象。"""

    backend_name: str = "abstract"

    @abstractmethod
    def get_user_sequence(self, user_id: int) -> List[int]:
        """返回用户最近点击过的视频 id 列表（原始 id，最新在前）。"""

    @abstractmethod
    def set_user_sequence(self, user_id: int, items: List[int], ttl: Optional[int] = None) -> None:
        """写入用户行为序列。"""

    def close(self) -> None:  # pragma: no cover - 默认无需释放
        return None


class InMemoryCacheClient(CacheClient):
    """进程内实现：用 dict 模拟，接口与 Redis 版完全一致。"""

    backend_name = "in-memory"

    def __init__(self) -> None:
        self._store: Dict[int, List[int]] = {}

    def get_user_sequence(self, user_id: int) -> List[int]:
        return list(self._store.get(int(user_id), []))

    def set_user_sequence(self, user_id: int, items: List[int], ttl: Optional[int] = None) -> None:
        # 内存版忽略 ttl（进程重启即失效，本地开发足够）
        self._store[int(user_id)] = list(items)

    def size(self) -> int:
        return len(self._store)


class RedisCacheClient(CacheClient):
    """真实 Redis 实现。"""

    backend_name = "redis"

    def __init__(self, url: str, default_ttl: int = 3600) -> None:
        import redis  # 延迟导入：没装 redis 客户端时也不影响内存版使用

        self._client = redis.from_url(url, decode_responses=True)
        self._default_ttl = default_ttl
        # 主动 ping 一次，尽早暴露连接问题（失败会被 build_cache 捕获并降级）
        self._client.ping()

    def get_user_sequence(self, user_id: int) -> List[int]:
        raw = self._client.get(f"{KEY_PREFIX}{int(user_id)}")
        if not raw:
            return []
        try:
            return [int(x) for x in json.loads(raw)]
        except (ValueError, TypeError):
            logger.warning("用户 %s 的行为序列缓存格式异常，已忽略", user_id)
            return []

    def set_user_sequence(self, user_id: int, items: List[int], ttl: Optional[int] = None) -> None:
        # 用 JSON 数组存：比 Redis List 更直观，且天然支持整体覆盖
        self._client.set(
            f"{KEY_PREFIX}{int(user_id)}",
            json.dumps([int(x) for x in items]),
            ex=ttl or self._default_ttl,
        )

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # pragma: no cover
            pass


def build_cache(redis_url: str = "", ttl: int = 3600) -> CacheClient:
    """根据配置构造缓存客户端；Redis 不可用时自动降级为内存实现。"""
    if redis_url:
        try:
            client = RedisCacheClient(redis_url, ttl)
            logger.info("缓存后端: Redis (%s)", redis_url)
            return client
        except Exception as e:  # 连接失败不应导致服务起不来
            logger.warning("连接 Redis 失败(%s)，自动降级为进程内内存缓存", e)
    logger.info("缓存后端: 进程内内存（未配置 REDIS_URL）")
    return InMemoryCacheClient()
