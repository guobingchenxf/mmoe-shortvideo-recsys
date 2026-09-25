"""短视频推荐场景的模拟数据生成脚本（Step 1）。

产出三张表到 data/raw/：
    users.csv         用户属性
    videos.csv        视频内容
    interactions.csv  曝光交互日志（含 3 个监督标签：click / like / watch_time_ratio）

=====================================================================
【为什么这样造数据 —— 这是整个项目的地基，值得解释清楚】
=====================================================================
真实短视频业务有三个关键特点，模拟数据必须复现它们，否则后续 MMoE 就学不到东西：

1) 用户兴趣有隐含结构：不同用户偏好不同垂类（有人爱看游戏、有人爱看美食），
   这种偏好不是显式给定的，而是"隐因子"，需要用模型去学。
2) 视频热度差异极大：头部视频曝光量远高于长尾（幂律/对数正态分布）。
3) 多个行为目标「相关但不同」：
   - 愿意点开的视频，往往也更容易看完、更容易点赞（相关性 => 共享专家有用）；
   - 但三者绝不等价（点击 ≠ 点赞，看完 ≠ 点赞），存在目标冲突
     （例如"标题党"能骗到点击，却拉低完播和点赞）。
   这正是多任务学习（MMoE）要解决的场景。

因此这里用「隐因子模型 + 截距校准」生成数据：
  - 用户隐兴趣向量 u_i、视频隐主题向量 v_j；
  - 相关性 r = <u_i, v_j> 决定点击/完播/点赞的基础 logit；
  - 再叠加视频热度先验，模拟曝光倾斜；
  - 最后用二分法求截距（bias），把 click / like 的全局比例**精确校准**到目标值。

【标签分布目标】（可在 configs/config.yaml 调整）
    click            ≈ 5%
    like             ≈ 1%
    watch_time_ratio  长尾分布（Beta 混合）

用法：
    python -m recsys.data.generate_data
    python -m recsys.data.generate_data --n-interactions 200000
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# 兼容两种运行方式：
#   python -m recsys.data.generate_data
#   python recsys/data/generate_data.py
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from recsys.config import Config, load_config
from recsys.data import schema
from recsys.utils.logger import get_logger

logger = get_logger("recsys.data.generate")

# 固定基准时间：避免使用 now()，保证多次运行生成的数据完全可复现
BASE_DATE = datetime(2026, 9, 1, 0, 0, 0)

# 一天中各小时的活跃权重（晚高峰明显，符合真实短视频产品的使用习惯）
_HOUR_WEIGHTS = np.array(
    [1, 1, 1, 1, 1, 2, 4, 7, 8, 7, 6, 7, 8, 7, 7, 8, 9, 12, 15, 18, 20, 16, 8, 3],
    dtype=np.float64,
)


# =====================================================================
# 工具函数
# =====================================================================
def _sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid。"""
    return 1.0 / (1.0 + np.exp(-x))


def _standardize(x: np.ndarray) -> np.ndarray:
    """z-score 标准化（防止除零）。"""
    return (x - x.mean()) / (x.std() + 1e-8)


def calibrate_intercept(
    logits: np.ndarray,
    target_rate: float,
    weights: np.ndarray | None = None,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> float:
    """二分法求偏置 b，使加权正例比例精确等于 target_rate。

    为什么需要它：sigmoid(logits) 的均值由 logits 的分布决定，随手设一个截距
    很难命中 5% / 1% 这种精确目标。二分法单调、稳定，200 次迭代足够收敛到 1e-6。

    Args:
        logits: 未加截距的 logit 数组。
        target_rate: 目标正例率，如 0.05。
        weights: 每个样本的计数权重（如 click 向量，用于计算"全曝光口径"的点赞率）。
                 None 表示全 1。
        tol: 收敛容差。
        max_iter: 最大迭代次数。

    Returns:
        标量偏置 b。
    """
    lo, hi = -30.0, 30.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        p = _sigmoid(logits + mid)
        rate = float(p.mean() if weights is None else (weights * p).mean())
        if abs(rate - target_rate) < tol:
            return mid
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# =====================================================================
# 三张表的生成
# =====================================================================
def generate_users(cfg: Config) -> pd.DataFrame:
    """生成用户属性表。"""
    n = cfg.data.n_users
    rng = np.random.default_rng(cfg.project.seed)

    users = pd.DataFrame(
        {
            schema.UserCol.USER_ID: np.arange(n, dtype=np.int64),
            # 年龄大致正态分布，裁剪到合理区间
            schema.UserCol.AGE: np.clip(rng.normal(28, 8, n), 12, 70).round().astype(np.int16),
            schema.UserCol.GENDER: rng.integers(0, 3, n).astype(np.int8),        # 0 女 / 1 男 / 2 未知
            schema.UserCol.CITY_LEVEL: rng.integers(1, 6, n).astype(np.int8),    # 1~5 线城市
            schema.UserCol.ACTIVE_LEVEL: rng.integers(0, 4, n).astype(np.int8),  # 活跃度分档
            schema.UserCol.REGISTER_DAYS: rng.integers(1, 1500, n).astype(np.int32),
        }
    )
    logger.info("用户表生成完成: %d 行", len(users))
    return users


def generate_videos(cfg: Config) -> pd.DataFrame:
    """生成视频内容表。"""
    n = cfg.data.n_videos
    rng = np.random.default_rng(cfg.project.seed + 1)

    # 热度服从对数正态：少数头部视频热度极高，长尾视频热度极低
    popularity = rng.lognormal(mean=0.0, sigma=0.8, size=n).round(4)

    # 每个视频挂 1~max_genre_len 个标签（multi-hot 特征的来源）
    tag_counts = rng.integers(1, cfg.features.max_genre_len + 1, size=n)
    genres = [
        schema.GENRE_SEP.join(
            map(str, np.sort(rng.choice(schema.N_TAG_VOCAB, size=int(c), replace=False)))
        )
        for c in tag_counts
    ]

    videos = pd.DataFrame(
        {
            schema.VideoCol.VIDEO_ID: np.arange(n, dtype=np.int64),
            schema.VideoCol.AUTHOR_ID: rng.integers(0, cfg.data.n_authors, n).astype(np.int32),
            schema.VideoCol.CATEGORY: rng.integers(0, cfg.data.n_categories, n).astype(np.int8),
            schema.VideoCol.GENRES: genres,
            # 短视频时长：5s ~ 120s
            schema.VideoCol.DURATION_MS: rng.integers(5_000, 120_000, n).astype(np.int32),
            schema.VideoCol.UPLOAD_DAYS_AGO: rng.integers(0, 365, n).astype(np.int32),
            schema.VideoCol.POPULARITY: popularity,
        }
    )
    logger.info("视频表生成完成: %d 行", len(videos))
    return videos


def _sample_timestamps(rng: np.random.Generator, n: int, n_days: int) -> pd.Series:
    """按"晚高峰"规律采样曝光时间，返回排好序的时间戳。"""
    day_offset = rng.integers(0, n_days, n)                       # 距今第几天
    hour = rng.choice(24, size=n, p=_HOUR_WEIGHTS / _HOUR_WEIGHTS.sum())
    minute = rng.integers(0, 60, n)
    second = rng.integers(0, 60, n)

    ts = [
        BASE_DATE
        - timedelta(days=int(d), hours=int(h), minutes=int(m), seconds=int(s))
        for d, h, m, s in zip(day_offset, hour, minute, second)
    ]
    return pd.Series(pd.to_datetime(ts)).sort_values(ignore_index=True)


def generate_interactions(
    users: pd.DataFrame, videos: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """生成曝光交互日志（含 3 个监督标签）。

    生成顺序遵循"因果"直觉：相关性 -> 观看行为(完播率) -> 点击 -> 点赞。
    其中 click 与 like 用截距校准精确命中目标比例。
    """
    n = cfg.data.n_interactions
    k = cfg.data.n_topics
    rng = np.random.default_rng(cfg.project.seed + 2)

    n_users, n_videos = len(users), len(videos)

    # ---------- 1. 隐因子：用户兴趣 & 视频主题 ----------
    # Dirichlet(0.3) 产生"稀疏偏好"——每个用户只对少数几个主题有兴趣，更贴近真实
    user_topic = rng.dirichlet(np.full(k, 0.3), size=n_users)     # (n_users, k)
    video_topic = rng.dirichlet(np.full(k, 0.3), size=n_videos)   # (n_videos, k)

    # ---------- 2. 曝光对采样：热门视频更容易被曝光 ----------
    popularity = videos[schema.VideoCol.POPULARITY].to_numpy()
    p_video = popularity / popularity.sum()                       # 曝光概率 ∝ 热度
    user_idx = rng.integers(0, n_users, n)
    video_idx = rng.choice(n_videos, size=n, p=p_video)

    # ---------- 3. 相关性：用户兴趣 与 视频主题 的内积 ----------
    relevance = np.einsum("ij,ij->i", user_topic[user_idx], video_topic[video_idx])
    rel_z = _standardize(relevance)
    pop_z = _standardize(np.log1p(popularity[video_idx]))

    # ---------- 4. 点击（校准到 target_ctr）----------
    click_logits = 1.6 * rel_z + 0.6 * pop_z
    click_bias = calibrate_intercept(click_logits, cfg.data.target_ctr)
    p_click = _sigmoid(click_logits + click_bias)
    click = (rng.random(n) < p_click).astype(np.int8)

    # ---------- 5. 完播率（长尾）----------
    # 均值由相关性决定，再叠加 Beta 噪声形成长尾（concentration 越小越两极分化）
    mean_ratio = np.clip(0.12 + 0.55 * _sigmoid(1.1 * rel_z + 0.25 * pop_z), 0.03, 0.97)
    conc = cfg.data.watch_beta_concentration
    watch_ratio = rng.beta(mean_ratio * conc, (1.0 - mean_ratio) * conc)
    watch_ratio = watch_ratio.round(4)

    # ---------- 6. 点赞（依赖相关性 + 完播率，体现"多目标相关但不同"）----------
    # 完播率越高越可能点赞 —— 这条依赖关系正是 MMoE 比"三个独立模型"更有优势的原因
    like_logits = 1.2 * rel_z + 1.0 * _standardize(watch_ratio) + 0.3 * pop_z
    # 用 click 作权重做校准：口径是"全曝光"的点赞率，且未点击不可能点赞
    like_bias = calibrate_intercept(like_logits, cfg.data.target_like_rate, weights=click)
    p_like = _sigmoid(like_logits + like_bias)
    like = ((click == 1) & (rng.random(n) < p_like)).astype(np.int8)

    # ---------- 7. 上下文特征 ----------
    timestamp = _sample_timestamps(rng, n, cfg.data.n_days)

    interactions = pd.DataFrame(
        {
            schema.InterCol.USER_ID: user_idx.astype(np.int64),
            schema.InterCol.VIDEO_ID: video_idx.astype(np.int64),
            schema.InterCol.TIMESTAMP: timestamp,
            schema.InterCol.CLICK: click,
            schema.InterCol.LIKE: like,
            schema.InterCol.WATCH_RATIO: watch_ratio.astype(np.float32),
            schema.InterCol.HOUR: timestamp.dt.hour.astype(np.int8),
            schema.InterCol.DOW: timestamp.dt.dayofweek.astype(np.int8),
            schema.InterCol.DEVICE: rng.integers(0, 3, n).astype(np.int8),
            schema.InterCol.SOURCE: rng.integers(0, 5, n).astype(np.int8),
        }
    )
    logger.info("交互表生成完成: %d 行", len(interactions))
    return interactions


# =====================================================================
# 落盘与统计
# =====================================================================
def _save(df: pd.DataFrame, path: Path) -> None:
    """统一用 UTF-8 落盘（Windows 默认 GBK，中文会出问题）。"""
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("已写出: %s  (%.2f MB)", path.name, path.stat().st_size / 1e6)


def summarize(interactions: pd.DataFrame) -> None:
    """打印标签分布，验证是否符合业务假设。"""
    click_rate = interactions[schema.InterCol.CLICK].mean()
    like_rate = interactions[schema.InterCol.LIKE].mean()
    # 点赞/点击 的口径换算，便于直观理解
    ctr_given = like_rate / max(click_rate, 1e-9)
    wr = interactions[schema.InterCol.WATCH_RATIO]

    logger.info("--- 标签分布校验 ---")
    logger.info("点击率 CTR          = %.4f  (目标 0.05)", click_rate)
    logger.info("点赞率 Like(全曝光) = %.4f  (目标 0.01)", like_rate)
    logger.info("点赞/点击           = %.4f", ctr_given)
    logger.info(
        "完播率 分位数 P10/P50/P90 = %.3f / %.3f / %.3f  (均值 %.3f)",
        wr.quantile(0.10), wr.quantile(0.50), wr.quantile(0.90), wr.mean(),
    )

    # 目标相关性：点击/点赞/完播之间应当"相关但不同"
    corr_cl = interactions[[schema.InterCol.CLICK, schema.InterCol.LIKE]].corr().iloc[0, 1]
    corr_cw = interactions[[schema.InterCol.CLICK, schema.InterCol.WATCH_RATIO]].corr().iloc[0, 1]
    corr_lw = interactions[[schema.InterCol.LIKE, schema.InterCol.WATCH_RATIO]].corr().iloc[0, 1]
    logger.info(
        "目标相关性 corr: click~like=%.3f  click~watch=%.3f  like~watch=%.3f",
        corr_cl, corr_cw, corr_lw,
    )
    logger.info("---------------------")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成短视频推荐模拟数据")
    parser.add_argument("--config", default=None, help="配置文件路径（默认 configs/config.yaml）")
    parser.add_argument("--n-interactions", type=int, default=None, help="覆盖曝光样本数")
    parser.add_argument("--out-dir", default=None, help="输出目录（默认 data/raw）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.n_interactions is not None:
        cfg.data.n_interactions = args.n_interactions

    out_dir = cfg.ensure_dir(args.out_dir or cfg.paths.data_raw)
    logger.info("数据将输出到: %s", out_dir)

    users = generate_users(cfg)
    videos = generate_videos(cfg)
    interactions = generate_interactions(users, videos, cfg)

    _save(users, out_dir / schema.USERS_FILE)
    _save(videos, out_dir / schema.VIDEOS_FILE)
    _save(interactions, out_dir / schema.INTERACTIONS_FILE)

    summarize(interactions)
    logger.info("Step 1 数据生成完成")


if __name__ == "__main__":
    main()
