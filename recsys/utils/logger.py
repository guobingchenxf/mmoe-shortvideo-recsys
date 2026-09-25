"""统一日志工具。

为什么不用 print：
- 训练脚本、数据生成脚本、在线服务都需要日志，且希望格式统一、可写文件、可分级；
- 直接用 logging 配置容易重复添加 handler（重复输出），这里做了幂等处理。
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str = "recsys",
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """获取（并惰性初始化）一个 logger。

    Args:
        name: logger 名称，建议用模块名，便于定位日志来源。
        level: 日志级别字符串，如 "INFO" / "DEBUG"。
        log_file: 可选，同时把日志写入该文件（UTF-8 编码）。

    Returns:
        已配置好 handler 的 logger；重复调用同一 name 不会重复添加 handler。
    """
    logger = logging.getLogger(name)

    # 幂等：已经配置过就直接返回，避免 handler 叠加导致日志重复打印
    if getattr(logger, "_recsys_configured", False):
        return logger

    logger.setLevel(level.upper())
    formatter = logging.Formatter(_LOG_FORMAT, _DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        # Windows 下必须显式指定 utf-8，否则中文日志会因系统默认 GBK 编码报错
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 关闭向 root logger 冒泡，防止与第三方库的日志配置相互干扰
    logger.propagate = False
    logger._recsys_configured = True  # type: ignore[attr-defined]
    return logger
