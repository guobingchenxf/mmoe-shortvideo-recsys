"""IO 工具：JSON 读写封装。

统一使用 UTF-8，避免 Windows 默认 GBK 造成中文内容损坏。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    """把对象序列化为 JSON 文件（UTF-8）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    return p


def load_json(path: str | Path) -> Any:
    """读取 JSON 文件（UTF-8）。"""
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)
