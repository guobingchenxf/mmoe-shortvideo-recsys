"""数据生成层单元测试。

本文件主要重点验证两件事：
1. 截距校准能否精确命中目标点击率/点赞率（这是数据可信度的基础）；
2. 生成的数据是否满足业务约束（点赞必须以点击为前提、完播率在 [0,1]）。
"""

from __future__ import annotations

import numpy as np

from recsys.config import load_config
from recsys.data import schema
from recsys.data.generate_data import (
    calibrate_intercept,
    generate_interactions,
    generate_users,
    generate_videos,
)


def test_calibrate_intercept_hits_target_rate() -> None:
    """二分法求出的截距，应使加权正例率精确等于目标值。"""
    rng = np.random.default_rng(0)
    logits = rng.normal(0.0, 1.0, size=20000)

    for target in (0.05, 0.01, 0.30, 0.50):
        bias = calibrate_intercept(logits, target)
        rate = float((1.0 / (1.0 + np.exp(-(logits + bias)))).mean())
        assert abs(rate - target) < 1e-5, f"target={target}, got={rate}"


def test_calibrate_intercept_with_weights() -> None:
    """带权重（如"仅点击样本才可能点赞"）时也应命中目标。"""
    rng = np.random.default_rng(1)
    logits = rng.normal(0.0, 1.0, size=20000)
    weights = (rng.random(20000) < 0.05).astype(np.float64)  # 模拟 click

    target = 0.01
    bias = calibrate_intercept(logits, target, weights=weights)
    p = 1.0 / (1.0 + np.exp(-(logits + bias)))
    assert abs(float((weights * p).mean()) - target) < 1e-5


def test_generated_data_satisfies_business_constraints() -> None:
    """小规模生成一次，校验标签分布与业务约束。"""
    cfg = load_config()
    cfg.data.n_users = 300
    cfg.data.n_videos = 500
    cfg.data.n_authors = 50
    cfg.data.n_interactions = 20000

    users = generate_users(cfg)
    videos = generate_videos(cfg)
    inter = generate_interactions(users, videos, cfg)

    # 行数与主键完整
    assert len(users) == 300 and len(videos) == 500
    assert len(inter) == 20000
    assert inter[schema.InterCol.USER_ID].between(0, 299).all()
    assert inter[schema.InterCol.VIDEO_ID].between(0, 499).all()

    # 标签分布命中目标（合成数据有随机性，故给一定容差）
    assert abs(inter[schema.InterCol.CLICK].mean() - cfg.data.target_ctr) < 0.005
    assert abs(inter[schema.InterCol.LIKE].mean() - cfg.data.target_like_rate) < 0.003

    # 业务约束：没有点击就不可能有点赞
    liked = inter[schema.InterCol.LIKE] == 1
    assert inter.loc[liked, schema.InterCol.CLICK].eq(1).all()

    # 完播率取值范围合法，且呈长尾（P50 < 均值说明分布右偏）
    ratio = inter[schema.InterCol.WATCH_RATIO]
    assert ratio.between(0.0, 1.0).all()
    assert ratio.quantile(0.5) < ratio.mean()

    # 暴露给模型的三个标签都是数值型且无缺失
    for col in schema.MULTI_TASK_LABELS:
        assert inter[col].notna().all()
