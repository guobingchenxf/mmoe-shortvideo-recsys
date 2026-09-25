"""特征列（FeatureColumn）声明模块。

=====================================================================
【设计动机 —— 为什么要有"特征声明层"】
=====================================================================
推荐系统的特征按模型侧处理方式可分为四类，处理逻辑完全不同：

    dense      连续数值  -> 归一化后直接拼进 MLP
    sparse     单值类别  -> 查 Embedding 表得到向量（如 user_id / category）
    multi_hot  多值类别  -> 多个 Embedding 求和池化（如视频的多个标签）
    sequence   变长序列  -> padding/截断到定长后做 Pooling 或 Attention

把"特征是什么"声明式地描述出来（而不是散落在各处写 if-else），带来三个好处：

1. Step 3 建模时，只依赖这份声明即可自动构建 Embedding 表，不用手写 8 个 nn.Embedding；
2. Step 4 在线服务时加载同一份声明，保证「离线训练特征」与「线上推理特征」口径完全一致
   —— 这是工业推荐系统最容易出线上事故的地方（训练/服务特征不一致，Training-Serving Skew）；
3. 新增特征只需在这里加一行，模型与服务的代码无需改动，符合"配置驱动"的工程实践。

约定：
- 变长特征（multi_hot / sequence）需要 padding，其 0 号位保留给 PAD，因此哈希编码结果从 1 开始；
- 单值特征（sparse）不需要 padding，索引 0 是合法取值（如 category=0）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from recsys.config import Config
from recsys.utils.io import load_json, save_json


class ColumnType(str, Enum):
    """特征类型。"""

    DENSE = "dense"
    SPARSE = "sparse"
    MULTI_HOT = "multi_hot"
    SEQUENCE = "sequence"


#: multi_hot / sequence 特征的 padding 索引。
#: 注意：sparse（单值）特征不保留该位，索引 0 是合法取值（如 category=0）。
PAD_INDEX = 0


# =====================================================================
# 特征名常量：Pipeline / Model / Service 三处共用，避免字符串写错
# =====================================================================
# ---- dense ----
F_AGE = "age"
F_DURATION = "duration_ms"
F_POPULARITY = "popularity"
F_UPLOAD_DAYS = "upload_days_ago"
F_REGISTER_DAYS = "register_days"
F_ACTIVE_LEVEL = "active_level"
F_CITY_LEVEL = "city_level"
F_HOUR_SIN = "hour_sin"
F_HOUR_COS = "hour_cos"
F_DOW_SIN = "dow_sin"
F_DOW_COS = "dow_cos"
F_USER_HIST_CTR = "user_hist_click_rate"
F_USER_HIST_CNT = "user_hist_cnt_log"

# ---- sparse ----
F_USER_ID = "user_id"
F_VIDEO_ID = "video_id"
F_AUTHOR_ID = "author_id"
F_CATEGORY = "category"
F_GENDER = "gender"
F_DEVICE = "device"
F_SOURCE = "source"
F_USER_CAT_CROSS = "user_cat_cross"

# ---- multi_hot ----
F_GENRES = "genres"

# ---- sequence ----
F_USER_SEQ = "user_seq"


@dataclass
class FeatureColumn:
    """单个特征的声明。

    Attributes:
        name: 特征名，同时也是 parquet 中的列名。
        ctype: 特征类型。
        vocab_size: 词表大小（Embedding 的 num_embeddings）；dense 无意义。
        max_len: multi_hot / sequence 的最大长度。
        log1p: dense 专用，是否先做 log1p 再标准化（用于强偏态特征）。
        desc: 业务含义说明。
    """

    name: str
    ctype: ColumnType
    vocab_size: int = 0
    max_len: int = 1
    log1p: bool = False
    desc: str = ""

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["ctype"] = self.ctype.value  # Enum -> str，保证 JSON 可序列化
        return d

    @staticmethod
    def from_dict(d: Dict) -> "FeatureColumn":
        d = dict(d)
        d["ctype"] = ColumnType(d["ctype"])
        return FeatureColumn(**d)


@dataclass
class FeatureSchema:
    """整套特征的声明集合，可持久化为 JSON 供离线/在线共用。"""

    version: str
    dense: List[FeatureColumn] = field(default_factory=list)
    sparse: List[FeatureColumn] = field(default_factory=list)
    multi_hot: List[FeatureColumn] = field(default_factory=list)
    sequence: List[FeatureColumn] = field(default_factory=list)

    # ---------- 便捷访问 ----------
    def columns(self) -> List[FeatureColumn]:
        """返回全部特征列（按 dense -> sparse -> multi_hot -> sequence 顺序）。"""
        return [*self.dense, *self.sparse, *self.multi_hot, *self.sequence]

    def column(self, name: str) -> FeatureColumn:
        """按名字取特征列，不存在则抛异常（快速失败优于静默出错）。"""
        for c in self.columns():
            if c.name == name:
                return c
        raise KeyError(f"特征 {name!r} 不在 FeatureSchema 中")

    @property
    def sparse_like(self) -> List[FeatureColumn]:
        """所有"需要查 Embedding 表"的特征（sparse + multi_hot + sequence）。"""
        return [*self.sparse, *self.multi_hot, *self.sequence]

    # ---------- 持久化 ----------
    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "dense": [c.to_dict() for c in self.dense],
            "sparse": [c.to_dict() for c in self.sparse],
            "multi_hot": [c.to_dict() for c in self.multi_hot],
            "sequence": [c.to_dict() for c in self.sequence],
        }

    def save(self, path: str | Path) -> Path:
        return save_json(self.to_dict(), path)

    @classmethod
    def from_dict(cls, d: Dict) -> "FeatureSchema":
        return cls(
            version=d.get("version", "unknown"),
            dense=[FeatureColumn.from_dict(x) for x in d.get("dense", [])],
            sparse=[FeatureColumn.from_dict(x) for x in d.get("sparse", [])],
            multi_hot=[FeatureColumn.from_dict(x) for x in d.get("multi_hot", [])],
            sequence=[FeatureColumn.from_dict(x) for x in d.get("sequence", [])],
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeatureSchema":
        return cls.from_dict(load_json(path))

    def summary(self) -> str:
        """人可读的特征清单，便于日志打印与自检。"""
        lines = [f"FeatureSchema {self.version}"]
        for title, cols in (
            ("DENSE", self.dense),
            ("SPARSE", self.sparse),
            ("MULTI_HOT", self.multi_hot),
            ("SEQUENCE", self.sequence),
        ):
            lines.append(f"  [{title}] {len(cols)} 个")
            for c in cols:
                extra = f"vocab={c.vocab_size}" if c.vocab_size else ""
                if c.max_len > 1:
                    extra += f" max_len={c.max_len}"
                if c.log1p:
                    extra += " log1p"
                lines.append(f"    - {c.name:<20s} {extra:<26s} {c.desc}")
        return "\n".join(lines)


def build_feature_schema(cfg: Config) -> FeatureSchema:
    """根据配置构造本项目使用的全部特征列。

    这里显式列出每一个特征及其业务含义，相当于一份"特征说明书"。
    """
    hb = cfg.features.hash_bucket_size

    dense = [
        FeatureColumn(F_AGE, ColumnType.DENSE, desc="用户年龄"),
        FeatureColumn(F_DURATION, ColumnType.DENSE, log1p=True, desc="视频时长(ms)，强偏态取log1p"),
        FeatureColumn(F_POPULARITY, ColumnType.DENSE, log1p=True, desc="视频热度，长尾取log1p"),
        FeatureColumn(F_UPLOAD_DAYS, ColumnType.DENSE, desc="视频上传距今天数(新鲜度)"),
        FeatureColumn(F_REGISTER_DAYS, ColumnType.DENSE, log1p=True, desc="用户注册天数(新老用户)"),
        FeatureColumn(F_ACTIVE_LEVEL, ColumnType.DENSE, desc="用户活跃度分档 0~3"),
        FeatureColumn(F_CITY_LEVEL, ColumnType.DENSE, desc="用户城市等级 1~5"),
        FeatureColumn(F_HOUR_SIN, ColumnType.DENSE, desc="小时的正弦编码(周期性)"),
        FeatureColumn(F_HOUR_COS, ColumnType.DENSE, desc="小时的余弦编码(周期性)"),
        FeatureColumn(F_DOW_SIN, ColumnType.DENSE, desc="星期的正弦编码(周期性)"),
        FeatureColumn(F_DOW_COS, ColumnType.DENSE, desc="星期的余弦编码(周期性)"),
        FeatureColumn(F_USER_HIST_CTR, ColumnType.DENSE, desc="截至当前时刻的用户历史点击率(防泄漏)"),
        FeatureColumn(F_USER_HIST_CNT, ColumnType.DENSE, log1p=True, desc="截至当前时刻的用户历史曝光次数"),
    ]

    sparse = [
        FeatureColumn(F_USER_ID, ColumnType.SPARSE, vocab_size=cfg.data.n_users, desc="用户 id"),
        FeatureColumn(F_VIDEO_ID, ColumnType.SPARSE, vocab_size=hb, desc="视频 id(哈希编码：短视频每日新增，词表不封闭)"),
        FeatureColumn(F_AUTHOR_ID, ColumnType.SPARSE, vocab_size=cfg.data.n_authors, desc="创作者 id"),
        FeatureColumn(F_CATEGORY, ColumnType.SPARSE, vocab_size=cfg.data.n_categories, desc="视频一级分类"),
        FeatureColumn(F_GENDER, ColumnType.SPARSE, vocab_size=3, desc="用户性别 0女/1男/2未知"),
        FeatureColumn(F_DEVICE, ColumnType.SPARSE, vocab_size=3, desc="设备类型"),
        FeatureColumn(F_SOURCE, ColumnType.SPARSE, vocab_size=5, desc="流量来源页面"),
        FeatureColumn(
            F_USER_CAT_CROSS, ColumnType.SPARSE, vocab_size=hb,
            desc="用户x分类 交叉特征(哈希编码)，显式提供特征交叉信号",
        ),
    ]

    multi_hot = [
        FeatureColumn(
            F_GENRES, ColumnType.MULTI_HOT, vocab_size=hb, max_len=cfg.features.max_genre_len,
            desc="视频标签(多值)，哈希编码后做 masked-mean 池化",
        ),
    ]

    sequence = [
        FeatureColumn(
            F_USER_SEQ, ColumnType.SEQUENCE, vocab_size=hb, max_len=cfg.data.max_seq_len,
            desc="用户最近点击过的视频序列(倒序，最新在前)，防泄漏",
        ),
    ]

    return FeatureSchema(version="v1", dense=dense, sparse=sparse, multi_hot=multi_hot, sequence=sequence)
