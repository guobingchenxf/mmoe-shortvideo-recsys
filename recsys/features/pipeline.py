"""离线特征工程流水线（Step 2）。

    raw CSV ──► 特征构造 ──► 时间切分 ──► 训练集拟合归一化 ──► parquet + 特征spec

=====================================================================
【三条铁律：也是推荐系统面试的高频考点】
=====================================================================
1) 无标签泄漏（Label Leakage）
   所有"历史统计"与"行为序列"特征，只使用当前样本时刻**之前**发生的数据。
   例如"用户历史点击率"必须排除当前这条样本本身的点击结果。

2) 无统计量泄漏（Statistics Leakage）
   归一化的 mean/std 只在训练集上 fit，再 transform 验证集。
   用全量数据 fit 会让验证集信息回流到训练，离线指标虚高、上线掉点。

3) 时间切分（Temporal Split）
   按时间先后切分，而非随机切分。线上永远是"用过去预测未来"，
   随机切分会让模型"偷看未来"，离线 AUC 会明显虚高。

产物：
    data/processed/train.parquet      训练集特征
    data/processed/valid.parquet      验证集特征
    data/processed/feature_spec.json  特征声明（Step 3/4 复用，保证线上线下一致）
    data/processed/dense_norm.json    dense 标准化统计量（仅由训练集拟合）

用法：
    python -m recsys.features.pipeline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from recsys.config import Config, load_config
from recsys.data import schema
from recsys.features import feature_column as fc
from recsys.features.feature_column import FeatureSchema, build_feature_schema
from recsys.features.preprocess import (
    DenseProcessor,
    build_user_history_stats,
    build_user_sequence,
    encode_multi_hot,
    hash_encode,
)
from recsys.utils.io import save_json
from recsys.utils.logger import get_logger

logger = get_logger("recsys.features.pipeline")

#: 时间切分比例：前 80% 训练，后 20% 验证
TRAIN_RATIO = 0.8
#: 用户历史点击率平滑所用的业务先验
HIST_PRIOR = 0.05
HIST_ALPHA = 5.0


# =====================================================================
# 1. 读取原始数据
# =====================================================================
def load_raw(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取 Step 1 生成的三张原始表。"""
    raw_dir = cfg.abs_path(cfg.paths.data_raw)
    users = pd.read_csv(raw_dir / schema.USERS_FILE)
    videos = pd.read_csv(raw_dir / schema.VIDEOS_FILE)
    inter = pd.read_csv(raw_dir / schema.INTERACTIONS_FILE)
    inter[schema.InterCol.TIMESTAMP] = pd.to_datetime(inter[schema.InterCol.TIMESTAMP])
    logger.info(
        "读取原始数据: users=%d, videos=%d, interactions=%d",
        len(users), len(videos), len(inter),
    )
    return users, videos, inter


# =====================================================================
# 2. 构造特征
# =====================================================================
def build_feature_frame(
    users: pd.DataFrame,
    videos: pd.DataFrame,
    inter: pd.DataFrame,
    cfg: Config,
    spec: FeatureSchema,
) -> pd.DataFrame:
    """把三张原始表加工成一张宽表（含全部特征列 + 标签）。"""
    df = inter.copy()

    # ---------- 2.1 关联用户属性 / 视频属性 ----------
    df = df.merge(users, on=schema.UserCol.USER_ID, how="left")
    df = df.merge(videos, on=schema.VideoCol.VIDEO_ID, how="left")
    miss = int(df[schema.VideoCol.AUTHOR_ID].isna().sum() + df[schema.UserCol.AGE].isna().sum())
    if miss:
        logger.warning("关联后存在 %d 个缺失值（理论上应为 0）", miss)

    # ---------- 2.2 周期性特征：小时 / 星期 用 sin-cos 编码 ----------
    # 直接用 0~23 的整数会让模型以为 23 点和 0 点"距离很远"，sin-cos 保留了环状相邻性
    hour = df[schema.InterCol.HOUR].to_numpy(dtype=np.float64)
    dow = df[schema.InterCol.DOW].to_numpy(dtype=np.float64)
    df[fc.F_HOUR_SIN] = np.sin(2 * np.pi * hour / 24.0)
    df[fc.F_HOUR_COS] = np.cos(2 * np.pi * hour / 24.0)
    df[fc.F_DOW_SIN] = np.sin(2 * np.pi * dow / 7.0)
    df[fc.F_DOW_COS] = np.cos(2 * np.pi * dow / 7.0)

    # ---------- 2.3 防泄漏的历史统计特征 ----------
    hist_rate, hist_cnt = build_user_history_stats(
        df,
        user_col=schema.InterCol.USER_ID,
        label_col=schema.InterCol.CLICK,
        time_col=schema.InterCol.TIMESTAMP,
        prior=HIST_PRIOR,
        alpha=HIST_ALPHA,
    )
    df[fc.F_USER_HIST_CTR] = hist_rate
    df[fc.F_USER_HIST_CNT] = hist_cnt

    # ---------- 2.4 防泄漏的用户行为序列 ----------
    seq_col = spec.column(fc.F_USER_SEQ)
    seq = build_user_sequence(
        df,
        user_col=schema.InterCol.USER_ID,
        item_col=schema.InterCol.VIDEO_ID,
        label_col=schema.InterCol.CLICK,
        time_col=schema.InterCol.TIMESTAMP,
        max_len=seq_col.max_len,
        bucket_size=seq_col.vocab_size,
    )

    # ---------- 2.5 编码 ----------
    hb = cfg.features.hash_bucket_size
    hash_cache: dict = {}

    # 先把原始 video_id 改名保留，再用哈希值占用 "video_id" 这一列
    df = df.rename(columns={schema.VideoCol.VIDEO_ID: "raw_video_id"})

    # (a) 单值类别：词表封闭 -> 直接用原值作为 Embedding 索引
    #     user_id / author_id / category / gender / device / source 均为已知有限词表
    for col in spec.sparse:
        if col.name in (fc.F_VIDEO_ID, fc.F_USER_CAT_CROSS):
            continue
        src = {
            fc.F_USER_ID: schema.UserCol.USER_ID,
            fc.F_AUTHOR_ID: schema.VideoCol.AUTHOR_ID,
            fc.F_CATEGORY: schema.VideoCol.CATEGORY,
            fc.F_GENDER: schema.UserCol.GENDER,
            fc.F_DEVICE: schema.InterCol.DEVICE,
            fc.F_SOURCE: schema.InterCol.SOURCE,
        }[col.name]
        df[col.name] = df[src].to_numpy(dtype=np.int64)

    # (b) video_id：短视频每天新增大量视频，词表不封闭 -> 哈希编码
    df[fc.F_VIDEO_ID] = hash_encode(df["raw_video_id"], hb, cache=hash_cache)

    # (c) 交叉特征 user x category：显式给模型一个"特征交叉"信号
    cross = df[schema.UserCol.USER_ID].astype(str) + "_" + df[schema.VideoCol.CATEGORY].astype(str)
    df[fc.F_USER_CAT_CROSS] = hash_encode(cross, hb, cache=hash_cache)

    # (d) 多值特征 genres -> 定长 id 矩阵（哈希编码）
    mh_col = spec.column(fc.F_GENRES)
    genres_mat = encode_multi_hot(
        df[schema.VideoCol.GENRES], bucket_size=mh_col.vocab_size, max_len=mh_col.max_len,
        cache=hash_cache,
    )

    # (e) 序列特征
    df[fc.F_USER_SEQ] = [row.tolist() for row in seq]
    df[fc.F_GENRES] = [row.tolist() for row in genres_mat]

    # 哈希碰撞诊断：桶数有限时，"不同原始值映射到同一桶"不可避免，量化一下影响
    _log_hash_collision("video_id", df["raw_video_id"].to_numpy(), hb)

    return df


def _log_hash_collision(name: str, raw_values: np.ndarray, bucket_size: int) -> None:
    """统计哈希碰撞率：不同原始值被映射到同一桶的比例。"""
    n_unique = len(np.unique(raw_values))
    distinct_buckets = len(np.unique(hash_encode(np.unique(raw_values), bucket_size)))
    collision = 1.0 - distinct_buckets / max(n_unique, 1)
    logger.info(
        "哈希碰撞诊断 [%s]: 唯一值=%d, 唯一桶=%d, 碰撞率=%.2f%% (桶数=%d)",
        name, n_unique, distinct_buckets, collision * 100, bucket_size,
    )


# =====================================================================
# 3. 校验
# =====================================================================
def validate_frame(df: pd.DataFrame, spec: FeatureSchema) -> None:
    """校验编码后的取值是否落在 Embedding 表范围内。"""
    for c in spec.sparse:
        v = df[c.name].to_numpy()
        assert v.min() >= 0, f"{c.name} 出现负索引"
        assert v.max() < c.vocab_size, f"{c.name} 索引越界: {v.max()} >= {c.vocab_size}"
    for c in spec.multi_hot + spec.sequence:
        arr = np.stack([np.asarray(x) for x in df[c.name]])
        assert arr.min() >= 0, f"{c.name} 出现负索引"
        assert arr.max() < c.vocab_size, f"{c.name} 索引越界: {arr.max()} >= {c.vocab_size}"
        assert arr.shape[1] == c.max_len, f"{c.name} 长度应为 {c.max_len}"
    logger.info("特征取值校验通过：所有索引均在 Embedding 范围内")


# =====================================================================
# 4. 主流程
# =====================================================================
def run(cfg: Config) -> None:
    spec = build_feature_schema(cfg)
    logger.info("特征声明如下:\n%s", spec.summary())

    users, videos, inter = load_raw(cfg)
    df = build_feature_frame(users, videos, inter, cfg, spec)

    # ---------- 时间切分 ----------
    df = df.sort_values(schema.InterCol.TIMESTAMP).reset_index(drop=True)
    split_idx = int(len(df) * TRAIN_RATIO)
    train, valid = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    logger.info(
        "时间切分: train=%d (%s ~ %s), valid=%d (%s ~ %s)",
        len(train), train[schema.InterCol.TIMESTAMP].min(), train[schema.InterCol.TIMESTAMP].max(),
        len(valid), valid[schema.InterCol.TIMESTAMP].min(), valid[schema.InterCol.TIMESTAMP].max(),
    )

    # ---------- 归一化：只在训练集上 fit ----------
    dense_proc = DenseProcessor(spec.dense).fit(train)
    logger.info("已在训练集上拟合 dense 归一化统计量: %d 个特征", len(dense_proc.feature_names))

    validate_frame(df, spec)

    # ---------- 落盘 ----------
    out_dir = cfg.ensure_dir(cfg.paths.data_processed)
    train.to_parquet(out_dir / "train.parquet", index=False)
    valid.to_parquet(out_dir / "valid.parquet", index=False)
    spec.save(out_dir / "feature_spec.json")
    save_json(dense_proc.state_dict(), out_dir / "dense_norm.json")

    logger.info("已写出: train.parquet / valid.parquet / feature_spec.json / dense_norm.json")
    logger.info("训练集正样本: click=%d (%.4f), like=%d (%.4f)",
                int(train[schema.InterCol.CLICK].sum()), train[schema.InterCol.CLICK].mean(),
                int(train[schema.InterCol.LIKE].sum()), train[schema.InterCol.LIKE].mean())
    logger.info("验证集正样本: click=%d (%.4f), like=%d (%.4f)",
                int(valid[schema.InterCol.CLICK].sum()), valid[schema.InterCol.CLICK].mean(),
                int(valid[schema.InterCol.LIKE].sum()), valid[schema.InterCol.LIKE].mean())
    logger.info("Step 2 特征工程完成")


def main() -> None:
    parser = argparse.ArgumentParser(description="离线特征工程流水线")
    parser.add_argument("--config", default=None, help="配置文件路径（默认 configs/config.yaml）")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
