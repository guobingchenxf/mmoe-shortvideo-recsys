"""特征工程层单元测试。

重中之重是**防泄漏**：历史统计与行为序列只能使用当前样本之前的信息。
这类 bug 不会让程序崩溃，只会让离线指标虚高、上线掉点，属于最难发现的一类问题，
所以必须有测试守住。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.data import schema
from recsys.features.feature_column import ColumnType, FeatureColumn
from recsys.features.preprocess import (
    DenseProcessor,
    build_user_history_stats,
    build_user_sequence,
    encode_multi_hot,
    hash_encode,
)


# =====================================================================
# 哈希编码
# =====================================================================
def test_hash_encode_is_stable_and_in_range() -> None:
    """同一输入必须稳定映射到同一桶，且结果落在 [1, bucket_size-1]。"""
    values = ["a", "b", "a", "3", "video_999"]
    out1 = hash_encode(values, bucket_size=1000)
    out2 = hash_encode(values, bucket_size=1000)

    assert np.array_equal(out1, out2), "同样输入两次编码结果不一致"
    assert out1.min() >= 1, "0 号位应保留给 PAD/UNK"
    assert out1.max() < 1000
    assert out1[0] == out1[2], "相同取值应映射到同一个桶"


def test_hash_encode_different_buckets_differ() -> None:
    """哈希结果依赖桶数，改桶数会整体改变索引空间（必须离线/在线一致的原因）。"""
    values = [str(i) for i in range(200)]
    small = hash_encode(values, bucket_size=100)
    large = hash_encode(values, bucket_size=1000)
    assert not np.array_equal(small, large)


# =====================================================================
# Dense 标准化
# =====================================================================
def test_dense_processor_standardizes() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(10, 3, 500), "y": np.abs(rng.normal(0, 5, 500))})
    cols = [
        FeatureColumn("x", ColumnType.DENSE),
        FeatureColumn("y", ColumnType.DENSE, log1p=True),
    ]
    proc = DenseProcessor(cols).fit(df)
    out = proc.transform(df)

    assert out.shape == (500, 2)
    assert abs(float(out[:, 0].mean())) < 1e-4
    assert abs(float(out[:, 0].std()) - 1.0) < 1e-3

    # log1p 后仍是标准化输出：均值 0 / 标准差 1
    assert abs(float(out[:, 1].mean())) < 1e-4
    assert abs(float(out[:, 1].std()) - 1.0) < 1e-3


def test_transform_one_matches_transform() -> None:
    """单条推理与批量推理必须给出完全一致的特征（Training-Serving 一致性）。"""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, 20.0, 30.0, 40.0]})
    cols = [
        FeatureColumn("x", ColumnType.DENSE),
        FeatureColumn("y", ColumnType.DENSE, log1p=True),
    ]
    proc = DenseProcessor(cols).fit(df)
    batch = proc.transform(df)

    for i in range(len(df)):
        one = proc.transform_one({"x": df["x"][i], "y": df["y"][i]})
        assert np.allclose(batch[i], one, atol=1e-6)


# =====================================================================
# Multi-hot
# =====================================================================
def test_encode_multi_hot_padding_and_truncation() -> None:
    out = encode_multi_hot(["a|b|c|d|e|f", "", "x"], bucket_size=100, max_len=3)

    assert out.shape == (3, 3)
    assert (out[0] > 0).all(), "超长时应截断到 max_len，且都是有效 id"
    assert (out[1] == 0).all(), "空值应全部填充为 PAD(0)"
    assert out[2, 0] > 0 and out[2, 1] == 0 and out[2, 2] == 0
    assert out.min() >= 0 and out.max() < 100


# =====================================================================
# 防泄漏：行为序列
# =====================================================================
def test_user_sequence_has_no_future_leakage() -> None:
    """第 i 行的序列只能包含它之前被点击过的视频，且最新在最前。"""
    df = pd.DataFrame({
        schema.InterCol.USER_ID: [1, 1, 1, 1],
        schema.InterCol.VIDEO_ID: [10, 20, 30, 40],
        schema.InterCol.TIMESTAMP: pd.to_datetime(
            ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
        ),
        schema.InterCol.CLICK: [1, 0, 1, 1],
    })

    seq = build_user_sequence(
        df,
        user_col=schema.InterCol.USER_ID,
        item_col=schema.InterCol.VIDEO_ID,
        label_col=schema.InterCol.CLICK,
        time_col=schema.InterCol.TIMESTAMP,
        max_len=5,
        bucket_size=1000,
    )
    h10 = int(hash_encode(["10"], 1000)[0])
    h30 = int(hash_encode(["30"], 1000)[0])

    # 第 0 行：没有任何历史
    assert (seq[0] == 0).all()

    # 第 1 行：历史只有第 0 行点击的 video 10
    assert seq[1][0] == h10 and seq[1][1] == 0

    # 第 2 行：第 1 行未点击，历史仍只有 video 10
    assert seq[2][0] == h10 and seq[2][1] == 0

    # 第 3 行：历史 = [最近点击的 30, 更早的 10]，且绝不包含自身 40
    assert seq[3][0] == h30 and seq[3][1] == h10 and seq[3][2] == 0


def test_user_sequence_respects_max_len_and_order() -> None:
    """序列长度受 max_len 约束，且保留最近的行为（不含当前样本自身）。"""
    n = 8
    df = pd.DataFrame({
        schema.InterCol.USER_ID: [1] * n,
        schema.InterCol.VIDEO_ID: list(range(100, 100 + n)),
        schema.InterCol.TIMESTAMP: pd.date_range("2026-01-01", periods=n, freq="D"),
        schema.InterCol.CLICK: [1] * n,
    })
    seq = build_user_sequence(
        df, schema.InterCol.USER_ID, schema.InterCol.VIDEO_ID,
        schema.InterCol.CLICK, schema.InterCol.TIMESTAMP, max_len=3, bucket_size=1000,
    )
    # 最后一行（video=107）此时已点击过 100~106，但只能看到"自己之前"的行为，
    # 且 max_len=3 只保留最近的 3 条 => [106, 105, 104]（最新在前），不含自身 107
    assert list(seq[-1][:3]) == list(hash_encode(["106", "105", "104"], 1000))
    assert seq.shape == (n, 3)


# =====================================================================
# 防泄漏：历史统计
# =====================================================================
def test_history_stats_excludes_current_row() -> None:
    """历史点击率必须排除当前样本自身的标签。"""
    df = pd.DataFrame({
        schema.InterCol.USER_ID: [1, 1, 1],
        schema.InterCol.TIMESTAMP: pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        schema.InterCol.CLICK: [1, 0, 1],
    })
    prior, alpha = 0.05, 5.0
    rate, cnt = build_user_history_stats(
        df, schema.InterCol.USER_ID, schema.InterCol.CLICK, schema.InterCol.TIMESTAMP,
        prior=prior, alpha=alpha,
    )

    # 第一行没有任何历史：计数 0，比率回落到先验
    assert cnt[0] == 0 and abs(float(rate[0]) - prior) < 1e-9
    # 第二行只知道第一行点击了（1 次点击 / 1 次曝光），不含自身
    assert cnt[1] == 1 and abs(float(rate[1]) - (1 + alpha * prior) / (1 + alpha)) < 1e-6
    assert cnt[2] == 2
