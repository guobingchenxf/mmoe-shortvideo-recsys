"""MMoE 训练流程：训练循环 + 验证指标 + EarlyStopping + 模型打包保存。

"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from recsys.config import Config, load_config
from recsys.data import schema
from recsys.features.feature_column import FeatureSchema
from recsys.features.preprocess import DenseProcessor
from recsys.models.dataset import RecTensors, build_tensors, make_loader
from recsys.models.loss import MultiTaskLoss
from recsys.models.metrics import compute_metrics
from recsys.models.mmoe import MMoE
from recsys.utils.io import load_json, save_json
from recsys.utils.logger import get_logger

logger = get_logger("recsys.models.train")

MODEL_FILE = "best_mmoe.pt"


# =====================================================================
# 工具
# =====================================================================
def set_seed(seed: int) -> None:
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch(batch: Dict, device: torch.device) -> Dict:
    """把 batch 里所有张量搬到目标设备。"""
    return {
        "dense": batch["dense"].to(device),
        "sparse": {k: v.to(device) for k, v in batch["sparse"].items()},
        "multi_hot": {k: v.to(device) for k, v in batch["multi_hot"].items()},
        "sequence": {k: v.to(device) for k, v in batch["sequence"].items()},
        "labels": {k: v.to(device) for k, v in batch["labels"].items()},
    }


def load_processed(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, FeatureSchema, DenseProcessor]:
    """读取 Step 2 产出的特征表、特征声明与归一化统计量。"""
    proc = cfg.abs_path(cfg.paths.data_processed)
    train_df = pd.read_parquet(proc / "train.parquet")
    valid_df = pd.read_parquet(proc / "valid.parquet")
    spec = FeatureSchema.load(proc / "feature_spec.json")
    dense_proc = DenseProcessor(spec.dense).load_state_dict(load_json(proc / "dense_norm.json"))
    logger.info("加载特征: train=%s, valid=%s", train_df.shape, valid_df.shape)
    return train_df, valid_df, spec, dense_proc


# =====================================================================
# 预测与评估
# =====================================================================
@torch.no_grad()
def predict(
    model: torch.nn.Module,
    tensors: RecTensors,
    label_names: Sequence[str],
    task_types: Dict[str, str],
    device: torch.device,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    """对全量数据推理，返回各任务的预测值（二分类已过 sigmoid）。"""
    model.eval()
    buf: Dict[str, List[np.ndarray]] = {n: [] for n in label_names}
    for batch in make_loader(tensors, batch_size, shuffle=False):
        b = move_batch(batch, device)
        logits = model(b)
        for i, name in enumerate(label_names):
            out = torch.sigmoid(logits[i]) if task_types[name] == "binary" else logits[i]
            buf[name].append(out.detach().cpu().numpy())
    return {n: np.concatenate(v) for n, v in buf.items()}


def evaluate(
    model: torch.nn.Module,
    tensors: RecTensors,
    label_names: Sequence[str],
    task_types: Dict[str, str],
    device: torch.device,
    batch_size: int,
    group: Optional[np.ndarray] = None,
    compute_gauc: bool = True,
) -> Dict[str, float]:
    """计算验证集指标（AUC / GAUC / RMSE）。

    GAUC 需要按用户分组循环计算，开销较大；训练过程中可只算 AUC/RMSE，
    等选出最优模型后再完整评估一次（见 train_model）。
    """
    preds = predict(model, tensors, label_names, task_types, device, batch_size)
    y_true = {n: tensors.labels[n].cpu().numpy() for n in label_names}
    return compute_metrics(
        label_names, task_types, y_true, preds,
        group=group if compute_gauc else None,
    )


def monitor_score(metrics: Dict[str, float], label_names: Sequence[str], task_types: Dict[str, str]) -> float:
    """把多个指标汇总成一个"越大越好"的标量，用于选最优模型与早停。

    策略：二分类任务取 AUC 的均值；若全是回归任务则取负 RMSE。
    """
    aucs = [
        metrics.get(f"{n}_auc")
        for n in label_names
        if task_types[n] == "binary"
    ]
    aucs = [a for a in aucs if a is not None and not np.isnan(a)]
    if aucs:
        return float(np.mean(aucs))
    rmses = [metrics.get(f"{n}_rmse", 0.0) for n in label_names]
    return -float(np.mean(rmses))


# =====================================================================
# 通用训练循环
# =====================================================================
def train_model(
    model: torch.nn.Module,
    train_t: RecTensors,
    valid_t: RecTensors,
    cfg: Config,
    label_names: Sequence[str],
    task_types: Dict[str, str],
    device: torch.device,
    group_valid: Optional[np.ndarray] = None,
    use_uncertainty: Optional[bool] = None,
    tag: str = "",
) -> Dict:
    """通用训练循环（MMoE 与单任务基线共用）。

    Returns:
        dict: 含最优 state_dict、最优指标、最优 epoch、训练历史、损失模块。
    """
    if use_uncertainty is None:
        # 只有多任务才启用不确定性加权；单任务没有"任务间平衡"的需求
        use_uncertainty = cfg.model.use_uncertainty_weighting and len(label_names) > 1

    model.to(device)
    criterion = MultiTaskLoss(label_names, task_types, use_uncertainty=use_uncertainty).to(device)

    # ---------- 优化器：稀疏参数与稠密参数分开更新 ----------
    # Embedding 表规模大（本项目 4 张 10 万级词表），若用普通 Adam 每步都要
    # 遍历整张表做动量更新，实测占单步耗时近一半。这里改为：
    #   Embedding -> 稀疏梯度 + SparseAdam（只更新本 batch 命中的行）
    #   其余参数  -> 普通 Adam（含损失函数里的可学习任务权重）
    sparse_params = model.sparse_parameters()
    dense_params = model.dense_parameters()
    use_sparse_opt = bool(sparse_params) and cfg.model.sparse_embeddings

    optimizers: List[torch.optim.Optimizer] = []
    if use_sparse_opt:
        # Embedding 用更大的学习率：每个 id 在一轮里只被更新几次，
        # 稠密层每步都更新；两者用同一个 lr 会导致 Embedding 学得过慢
        optimizers.append(torch.optim.SparseAdam(sparse_params, lr=cfg.model.embedding_lr))
        rest_params = dense_params
    else:
        rest_params = dense_params + sparse_params
    optimizers.append(
        torch.optim.Adam(
            rest_params + list(criterion.parameters()),
            lr=cfg.model.lr,
            weight_decay=cfg.model.weight_decay,
        )
    )
    logger.info(
        "%s优化器: %s | 稀疏参数=%d 稠密参数=%d",
        f"{tag} " if tag else "",
        "SparseAdam + Adam" if use_sparse_opt else "Adam",
        sum(p.numel() for p in sparse_params),
        sum(p.numel() for p in dense_params),
    )

    best_score, best_epoch, best_metrics, best_state = -float("inf"), -1, {}, None
    patience_left = cfg.model.patience
    history: List[Dict] = []

    for epoch in range(1, cfg.model.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, n_batch = 0.0, 0
        for batch in make_loader(train_t, cfg.model.batch_size, shuffle=True):
            b = move_batch(batch, device)
            logits = model(b)
            loss, _ = criterion(logits, b["labels"])
            for opt in optimizers:
                opt.zero_grad()
            loss.backward()
            for opt in optimizers:
                opt.step()
            total_loss += float(loss.detach())
            n_batch += 1

        # 训练过程中只算 AUC / RMSE（GAUC 开销大，留到最后对最优模型算一次）
        metrics = evaluate(
            model, valid_t, label_names, task_types, device, cfg.model.batch_size,
            group_valid, compute_gauc=False,
        )
        score = monitor_score(metrics, label_names, task_types)
        weights = criterion.learned_weights()

        record = {"epoch": epoch, "train_loss": total_loss / max(n_batch, 1), "monitor": score, **metrics}
        history.append(record)
        logger.info(
            "%s[epoch %2d] loss=%.4f | %s | monitor=%.4f | %.1fs | 任务权重=%s",
            f"{tag} " if tag else "",
            epoch,
            record["train_loss"],
            " ".join(
                f"{k}={v:.4f}" for k, v in metrics.items() if not np.isnan(v)
            ),
            score,
            time.time() - t0,
            {k: round(v, 3) for k, v in weights.items()},
        )

        if score > best_score + 1e-6:
            best_score, best_epoch = score, epoch
            best_metrics = dict(metrics)
            # 深拷贝到 CPU，避免后续训练覆盖最优权重
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.model.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("%s早停：验证集指标连续 %d 轮未提升", f"{tag} " if tag else "", cfg.model.patience)
                break

    assert best_state is not None, "训练未产生任何有效模型"
    model.load_state_dict(best_state)

    # 用最优权重完整评估一次（含 GAUC）
    best_metrics = evaluate(
        model, valid_t, label_names, task_types, device, cfg.model.batch_size,
        group_valid, compute_gauc=True,
    )
    logger.info("%s最优模型完整指标(含GAUC): %s",
                f"{tag} " if tag else "", {k: round(v, 4) for k, v in best_metrics.items()})

    return {
        "state_dict": best_state,
        "metrics": best_metrics,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "history": history,
        "criterion": criterion,
    }


def load_bundle(path: str | Path, device: torch.device | None = None):
    """加载自包含模型包，返回 (model, spec, dense_proc, bundle)。

    供离线评估（evaluate.py）与在线服务（services/）复用，
    保证"训练时用的特征口径"与"推理时用的特征口径"来自同一份产物。
    """
    bundle = torch.load(Path(path), map_location="cpu", weights_only=False)
    spec = FeatureSchema.from_dict(bundle["feature_spec"])
    dense_proc = DenseProcessor(spec.dense).load_state_dict(bundle["dense_norm"])
    model = MMoE(spec=spec, **bundle["model_kwargs"])
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    if device is not None:
        model.to(device)
    return model, spec, dense_proc, bundle


def build_mmoe(cfg: Config, spec: FeatureSchema, num_tasks: int, num_experts: Optional[int] = None) -> MMoE:
    """按配置构造 MMoE。

    把 num_experts 设为 1 时，门控退化为恒等权重，模型等价于
    「Shared Bottom + 多塔」——这正是 MMoE 论文中的对照基线。
    """
    m = cfg.model
    return MMoE(
        spec=spec,
        num_tasks=num_tasks,
        embedding_dim=m.embedding_dim,
        sparse_embeddings=m.sparse_embeddings,
        shared_bottom_hidden=m.shared_bottom_hidden,
        num_experts=m.num_experts if num_experts is None else num_experts,
        expert_hidden=m.expert_hidden,
        gate_hidden=m.gate_hidden,
        tower_hidden=m.tower_hidden,
        dropout=m.dropout,
    )


# =====================================================================
# 主流程
# =====================================================================
def run(cfg: Config, args) -> None:
    set_seed(cfg.project.seed)
    device = get_device()
    logger.info("使用设备: %s", device)

    if args.epochs:
        cfg.model.epochs = args.epochs

    train_df, valid_df, spec, dense_proc = load_processed(cfg)
    train_t = build_tensors(train_df, spec, dense_proc)
    valid_t = build_tensors(valid_df, spec, dense_proc)

    if args.sample:
        rng = np.random.default_rng(cfg.project.seed)
        train_t = train_t.subset(np.sort(rng.choice(train_t.n, min(args.sample, train_t.n), replace=False)))
        logger.info("快速模式：训练集抽样至 %d 条", train_t.n)

    label_names = list(schema.MULTI_TASK_LABELS)
    task_types = dict(schema.LABEL_TASK_TYPES)

    model = build_mmoe(cfg, spec, num_tasks=len(label_names))
    logger.info(model.describe())

    result = train_model(
        model, train_t, valid_t, cfg, label_names, task_types, device,
        group_valid=valid_df[schema.InterCol.USER_ID].to_numpy(),
    )

    # ---------- 保存自包含 bundle ----------
    out_dir = cfg.ensure_dir(cfg.paths.model_dir)
    bundle = {
        "model_type": "mmoe",
        "state_dict": result["state_dict"],
        "feature_spec": spec.to_dict(),
        "dense_norm": dense_proc.state_dict(),
        "model_kwargs": {
            "num_tasks": len(label_names),
            "embedding_dim": cfg.model.embedding_dim,
            "sparse_embeddings": cfg.model.sparse_embeddings,
            "shared_bottom_hidden": list(cfg.model.shared_bottom_hidden),
            "num_experts": cfg.model.num_experts,
            "expert_hidden": list(cfg.model.expert_hidden),
            "gate_hidden": list(cfg.model.gate_hidden),
            "tower_hidden": list(cfg.model.tower_hidden),
            "dropout": cfg.model.dropout,
        },
        "label_names": label_names,
        "task_types": task_types,
        "metrics": result["metrics"],
        "best_epoch": result["best_epoch"],
    }
    ckpt_path = out_dir / MODEL_FILE
    torch.save(bundle, ckpt_path)

    # ---------- 保存训练历史 ----------
    log_dir = cfg.ensure_dir(cfg.paths.log_dir)
    pd.DataFrame(result["history"]).to_csv(log_dir / "train_history.csv", index=False, encoding="utf-8")
    save_json({"best": result["metrics"], "best_epoch": result["best_epoch"]}, log_dir / "train_best.json")

    logger.info("最优模型保存在: %s (epoch=%d)", ckpt_path, result["best_epoch"])
    logger.info("最优验证指标: %s", {k: round(v, 4) for k, v in result["metrics"].items()})
    logger.info("Step 3 MMoE 训练完成")


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 MMoE 多目标模型")
    parser.add_argument("--config", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置中的 epoch 数")
    parser.add_argument("--sample", type=int, default=None, help="只用 N 条训练样本（快速自检用）")
    args = parser.parse_args()
    run(load_config(args.config), args)


if __name__ == "__main__":
    main()
