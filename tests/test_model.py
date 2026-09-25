"""模型层单元测试：MMoE 结构正确性、多任务损失、评估指标。

这些测试用小词表 + 小网络，秒级跑完，用于防止重构时把模型结构改坏。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pytest
import torch

from recsys.config import load_config
from recsys.features.feature_column import FeatureSchema, build_feature_schema
from recsys.models.loss import MultiTaskLoss
from recsys.models.metrics import auc, gauc, rmse
from recsys.models.mmoe import MMoE, SingleTaskModel


# =====================================================================
# 测试夹具
# =====================================================================
def _small_setup() -> tuple[FeatureSchema, dict]:
    """构造一个小词表的特征声明与对应的模型超参。"""
    cfg = load_config()
    cfg.data.n_users = 50
    cfg.data.n_videos = 100
    cfg.data.n_authors = 20
    cfg.features.hash_bucket_size = 1000
    spec = build_feature_schema(cfg)
    kwargs = dict(
        embedding_dim=4,
        shared_bottom_hidden=[16],
        num_experts=4,
        expert_hidden=[16],
        gate_hidden=[],
        tower_hidden=[8],
        dropout=0.0,
    )
    return spec, kwargs


def _dummy_batch(spec: FeatureSchema, n: int = 8) -> Dict:
    return {
        "dense": torch.randn(n, len(spec.dense)),
        "sparse": {c.name: torch.randint(0, c.vocab_size, (n,)) for c in spec.sparse},
        "multi_hot": {
            c.name: torch.randint(0, c.vocab_size, (n, c.max_len)) for c in spec.multi_hot
        },
        "sequence": {
            c.name: torch.randint(0, c.vocab_size, (n, c.max_len)) for c in spec.sequence
        },
        "labels": {},
    }


# =====================================================================
# MMoE 结构
# =====================================================================
def test_mmoe_output_shape_and_count() -> None:
    spec, kwargs = _small_setup()
    model = MMoE(spec=spec, num_tasks=3, sparse_embeddings=True, **kwargs)

    out = model(_dummy_batch(spec, n=8))
    assert len(out) == 3, "任务塔数量必须等于任务数"
    assert all(o.shape == (8,) for o in out)


def test_gate_weights_sum_to_one() -> None:
    """门控输出必须是 softmax 归一化的权重分布。"""
    spec, kwargs = _small_setup()
    model = MMoE(spec=spec, num_tasks=3, sparse_embeddings=True, **kwargs)

    x = torch.randn(5, model.input_layer.output_dim)
    h = model.shared_bottom(x)
    for gate in model.gates:
        w = gate(h)
        assert w.shape == (5, 4)
        assert torch.allclose(w.sum(dim=-1), torch.ones(5), atol=1e-5)
        assert (w >= 0).all()


def test_sparse_and_dense_parameters_are_partitioned() -> None:
    """稀疏参数与稠密参数必须无重叠且并集为全部参数（否则优化器会漏更新/重复更新）。"""
    spec, kwargs = _small_setup()
    model = MMoE(spec=spec, num_tasks=3, sparse_embeddings=True, **kwargs)

    sparse_ids = {id(p) for p in model.sparse_parameters()}
    dense_ids = {id(p) for p in model.dense_parameters()}

    assert sparse_ids and dense_ids
    assert not (sparse_ids & dense_ids)
    assert len(sparse_ids) + len(dense_ids) == len(list(model.parameters()))


def test_sparse_embedding_produces_sparse_gradient() -> None:
    """开启 sparse=True 后，Embedding 的梯度必须是稀疏张量（否则 SparseAdam 会报错）。"""
    spec, kwargs = _small_setup()
    model = MMoE(spec=spec, num_tasks=3, sparse_embeddings=True, **kwargs)

    out = model(_dummy_batch(spec, n=4))
    sum(o.sum() for o in out).backward()

    emb = model.input_layer.sparse_emb[spec.sparse[0].name]
    assert emb.weight.grad is not None
    assert emb.weight.grad.is_sparse, "Embedding 梯度应为稀疏张量"


def test_single_task_model_returns_single_output() -> None:
    """单任务基线返回长度为 1 的列表，以便复用同一套训练/评估代码。"""
    spec, kwargs = _small_setup()
    model = SingleTaskModel(
        spec=spec, embedding_dim=4, sparse_embeddings=True,
        hidden=[16], dropout=0.0,
    )
    out = model(_dummy_batch(spec, n=6))
    assert len(out) == 1 and out[0].shape == (6,)


# =====================================================================
# 多任务损失
# =====================================================================
def test_multitask_loss_uncertainty_weighting_is_learnable() -> None:
    loss_fn = MultiTaskLoss(
        ["click", "watch"], {"click": "binary", "watch": "regression"}, use_uncertainty=True
    )
    logits = [torch.randn(10, requires_grad=True), torch.randn(10, requires_grad=True)]
    labels = {"click": torch.randint(0, 2, (10,)).float(), "watch": torch.rand(10)}

    total, per_task = loss_fn(logits, labels)
    total.backward()

    assert torch.isfinite(total)
    assert set(per_task) == {"click", "watch"}
    # 任务权重是可学习参数，必须收到梯度
    assert loss_fn.log_vars.grad is not None
    assert torch.isfinite(loss_fn.log_vars.grad).all()


def test_multitask_loss_fixed_weights() -> None:
    loss_fn = MultiTaskLoss(
        ["a", "b"], {"a": "binary", "b": "binary"},
        use_uncertainty=False, manual_weights=[2.0, 1.0],
    )
    logits = [torch.zeros(4), torch.zeros(4)]
    labels = {"a": torch.zeros(4), "b": torch.zeros(4)}
    total, _ = loss_fn(logits, labels)

    # logit=0 时 BCE = ln2，加权后应为 2*ln2 + 1*ln2 = 3*ln2
    assert abs(float(total) - 3 * float(np.log(2))) < 1e-5
    weights = loss_fn.learned_weights()
    assert abs(weights["a"] - 2.0) < 1e-6 and abs(weights["b"] - 1.0) < 1e-6


# =====================================================================
# 评估指标
# =====================================================================
def test_auc_perfect_and_random() -> None:
    y = np.array([0, 0, 1, 1])
    assert auc(y, np.array([0.1, 0.2, 0.3, 0.4])) == pytest.approx(1.0)
    assert auc(y, np.array([0.4, 0.3, 0.2, 0.1])) == pytest.approx(0.0)
    # 完全并列时 AUC 应退化为 0.5
    assert auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)


def test_auc_handles_single_class() -> None:
    """只有一类样本时 AUC 无定义，应返回 NaN 而不是抛异常。"""
    assert np.isnan(auc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])))


def test_gauc_skips_single_class_groups() -> None:
    """组内只有一类样本的用户应被跳过，不参与加权（否则 AUC 无意义）。"""
    y = np.array([0, 1, 1, 1])
    s = np.array([0.1, 0.9, 0.5, 0.6])
    g = np.array([1, 1, 2, 2])  # 用户 2 内只有正样本

    assert gauc(y, s, g) == pytest.approx(1.0)


def test_gauc_weighted_by_group_size() -> None:
    """GAUC 是按组内样本量加权平均，而非简单平均。"""
    y = np.array([0, 1, 1, 0, 1, 0])
    s = np.array([0.1, 0.9, 0.9, 0.1, 0.9, 0.1])
    g = np.array([1, 1, 2, 2, 2, 2])

    # 用户 1：AUC=1.0（2 条）；用户 2：AUC=1.0（4 条）→ 加权后仍为 1.0
    assert gauc(y, s, g) == pytest.approx(1.0)

    # 把用户 2 的顺序打乱（AUC=0），加权结果应为 (1.0*2 + 0.0*4) / 6
    s2 = np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
    assert gauc(y, s2, g) == pytest.approx(2.0 / 6.0)


def test_rmse() -> None:
    assert rmse(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == pytest.approx(0.0)
    assert rmse(np.array([1.0, 1.0]), np.array([2.0, 2.0])) == pytest.approx(1.0)
    assert rmse(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == pytest.approx(np.sqrt(12.5))
