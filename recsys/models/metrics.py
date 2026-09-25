"""推荐系统离线评估指标（纯 numpy 实现，不依赖 sklearn）。

为什么自己实现：
- 便于理解指标内部机制（面试常被追问 AUC 的物理含义、GAUC 为什么要按用户分组）；
- 推荐系统常用的 GAUC 在 sklearn 里并没有现成实现。

指标说明：
    AUC    全体样本的排序能力。随机猜为 0.5。
    GAUC   按用户分组算 AUC 再加权平均，更贴近推荐场景的排序质量。
           （推荐是"对同一个用户，把他更可能喜欢的排前面"，跨用户比较意义不大）
    RMSE   回归任务（完播率）的误差。
"""

from __future__ import annotations

import numpy as np


def _rank_average(x: np.ndarray) -> np.ndarray:
    """平均秩（并列取平均），等价于 scipy.stats.rankdata 的默认行为，但为向量化实现。"""
    sorter = np.argsort(x, kind="mergesort")
    inv = np.empty(len(x), dtype=np.int64)
    inv[sorter] = np.arange(len(x), dtype=np.int64)

    x_sorted = x[sorter]
    # obs[i] 为 True 表示第 i 个位置是一个"新分值"分组的起点
    obs = np.r_[True, x_sorted[1:] != x_sorted[:-1]]
    dense = obs.cumsum()[inv]
    # 每个分组的边界与大小
    count = np.r_[np.nonzero(obs)[0], len(x)]
    # 平均秩 = (组内最小秩 + 最大秩) / 2 = (count[d-1]+1 + count[d]) / 2
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC（等价的 Mann-Whitney U 统计量算法，O(n log n)）。

    AUC 的物理含义：随机取一个正样本和一个负样本，模型给正样本打分更高的概率。
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()

    n_pos = float(y_true.sum())
    n_neg = float(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")  # 只有一类样本时 AUC 无定义

    ranks = _rank_average(y_score)
    sum_ranks_pos = ranks[y_true == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def gauc(y_true: np.ndarray, y_score: np.ndarray, group: np.ndarray) -> float:
    """Group AUC：按用户分组计算 AUC，再以「组内曝光数」为权重加权平均。

    为什么推荐系统更看重 GAUC：
    AUC 是全局排序指标，会混入"用户之间活跃度差异"带来的偏差；
    而线上排序实际发生在「同一个用户的候选集内」，所以按用户分组评估更贴近真实效果。
    只统计组内同时含正负样本的用户（否则该组 AUC 无定义）。
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    group = np.asarray(group).ravel()

    weighted_sum, weight_total = 0.0, 0.0
    for g in np.unique(group):
        mask = group == g
        gt, gs = y_true[mask], y_score[mask]
        n_pos = gt.sum()
        if n_pos == 0 or n_pos == len(gt):
            continue  # 该用户没有区分度，跳过
        a = auc(gt, gs)
        if np.isnan(a):
            continue
        w = len(gt)  # 权重 = 该用户的曝光数
        weighted_sum += a * w
        weight_total += w
    return float(weighted_sum / weight_total) if weight_total > 0 else float("nan")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方根误差。"""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_metrics(
    label_names: tuple,
    task_types: dict,
    y_true: dict,
    y_pred: dict,
    group: np.ndarray | None = None,
) -> dict:
    """一次性计算所有任务的指标。

    Args:
        label_names: 标签名元组（顺序与模型输出一致）。
        task_types: {标签名: "binary" | "regression"}。
        y_true: {标签名: 真实值数组}。
        y_pred: {标签名: 预测值数组}（二分类传概率，回归传预测值）。
        group: 分组列（通常为 user_id），用于计算 GAUC。

    Returns:
        扁平字典，例如 {"click_auc": 0.72, "click_gauc": 0.70, ...}
    """
    out: dict = {}
    for name in label_names:
        t = np.asarray(y_true[name]).ravel()
        p = np.asarray(y_pred[name]).ravel()
        if task_types[name] == "binary":
            out[f"{name}_auc"] = auc(t, p)
            if group is not None:
                out[f"{name}_gauc"] = gauc(t, p, group)
        else:
            out[f"{name}_rmse"] = rmse(t, p)
    return out
