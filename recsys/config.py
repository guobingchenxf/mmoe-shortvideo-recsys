"""配置加载器：把 configs/config.yaml 解析为强类型 dataclass。


"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

import yaml

# 本文件位于 <project_root>/recsys/config.py，parents[1] 即项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

T = TypeVar("T")


@dataclass
class ProjectCfg:
    name: str = "mmoe-shortvideo-recsys"
    seed: int = 42


@dataclass
class PathsCfg:
    """各类产物目录（相对项目根）。"""

    data_raw: str = "data/raw"
    data_processed: str = "data/processed"
    model_dir: str = "artifacts/models"
    log_dir: str = "artifacts/logs"


@dataclass
class DataCfg:
    """数据生成相关配置。"""

    n_users: int = 5000
    n_videos: int = 20000
    n_interactions: int = 120000
    n_topics: int = 8
    n_categories: int = 20
    n_authors: int = 2000
    n_days: int = 30
    target_ctr: float = 0.05
    target_like_rate: float = 0.01
    watch_beta_concentration: float = 2.0
    max_seq_len: int = 50


@dataclass
class FeatureCfg:
    """特征工程相关配置。"""

    hash_bucket_size: int = 100000
    max_genre_len: int = 5


@dataclass
class ModelCfg:
    """MMoE 模型与训练超参。"""

    embedding_dim: int = 16
    sparse_embeddings: bool = True
    embedding_lr: float = 1e-2
    shared_bottom_hidden: List[int] = field(default_factory=lambda: [256])
    num_experts: int = 8
    expert_hidden: List[int] = field(default_factory=lambda: [128, 64])
    gate_hidden: List[int] = field(default_factory=list)
    tower_hidden: List[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 1024
    epochs: int = 20
    patience: int = 3
    weight_decay: float = 1e-5
    use_uncertainty_weighting: bool = True


@dataclass
class ServiceCfg:
    """在线服务配置。"""

    host: str = "0.0.0.0"
    port: int = 8000
    top_k: int = 10
    recall_size: int = 500
    redis_url: str = ""
    cache_ttl: int = 3600


@dataclass
class Config:
    """全局配置聚合对象。"""

    project: ProjectCfg = field(default_factory=ProjectCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)
    data: DataCfg = field(default_factory=DataCfg)
    features: FeatureCfg = field(default_factory=FeatureCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    service: ServiceCfg = field(default_factory=ServiceCfg)

    # ------------------------------------------------------------------
    # 路径工具：统一把「相对项目根的路径」解析成绝对路径
    # ------------------------------------------------------------------
    def abs_path(self, rel: str) -> Path:
        """相对路径 -> 绝对路径（传入绝对路径则原样返回）。"""
        p = Path(rel)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def ensure_dir(self, rel: str) -> Path:
        """解析路径并确保对应目录存在，返回绝对路径。"""
        p = self.abs_path(rel)
        p.mkdir(parents=True, exist_ok=True)
        return p


def _build(cls: Type[T], data: Optional[Dict[str, Any]]) -> T:
    """用 dict 中「该类已声明的字段」构造 dataclass，忽略多余键。

    这样 config.yaml 里可以自由添加实验性字段而不会导致加载失败。
    """
    data = data or {}
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    return cls(**{k: v for k, v in data.items() if k in known})  # type: ignore[call-arg]


def load_config(path: Optional[str | Path] = None) -> Config:
    """读取 YAML 配置并构造 Config 对象。

    Args:
        path: 配置文件路径；缺省使用 configs/config.yaml。

    Returns:
        Config: 强类型配置对象。
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    # 必须显式 encoding="utf-8"：Windows 默认使用 GBK，会让中文注释解码失败
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return Config(
        project=_build(ProjectCfg, raw.get("project")),
        paths=_build(PathsCfg, raw.get("paths")),
        data=_build(DataCfg, raw.get("data")),
        features=_build(FeatureCfg, raw.get("features")),
        model=_build(ModelCfg, raw.get("model")),
        service=_build(ServiceCfg, raw.get("service")),
    )


if __name__ == "__main__":
    # 快速自检：python -m recsys.config
    print(load_config())
