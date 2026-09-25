"""离线评估与基线对比（Step 3 的"结论"产出）。

=====================================================================
【实验设计：用数据回答两个问题】
=====================================================================
同一份数据、同一套特征、同一套训练配置（lr / batch / epochs / patience），只改模型结构：

  方案 A：单任务独立建模 ×3
          每个目标训练一个独立模型（各有一套 Embedding 与塔），任务之间零共享。
  方案 B：Shared-Bottom 多任务
          所有任务共享一个底层 MLP，各自接一个塔。这是 MMoE 论文里的对照基线。
          实现上等价于"只有 1 个专家"的 MMoE（门控 softmax 恒为 1）。
  方案 C：MMoE（多专家 + 多门控）—— 本项目方案

回答的问题：
  Q1 多任务学习相比"每个目标单独建模"是否有收益？   （B、C vs A）
  Q2 "多专家 + 门控"相比"硬共享底层"是否有收益？    （C vs B）

用法：
    python -m recsys.models.evaluate
    python -m recsys.models.evaluate --epochs 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from recsys.config import Config, load_config
from recsys.data import schema
from recsys.models.dataset import build_tensors
from recsys.models.mmoe import SingleTaskModel
from recsys.models.train import (
    MODEL_FILE,
    build_mmoe,
    evaluate,
    get_device,
    load_bundle,
    load_processed,
    set_seed,
    train_model,
)
from recsys.utils.io import save_json
from recsys.utils.logger import get_logger

logger = get_logger("recsys.models.evaluate")


def _train_baselines(
    cfg: Config,
    spec,
    train_t,
    valid_t,
    label_names: List[str],
    task_types: Dict[str, str],
    device: torch.device,
    group_valid: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """训练并评估各基线模型，返回 {方案名: 指标字典}。"""
    results: Dict[str, Dict[str, float]] = {}

    # ---------- 方案 A：3 个独立的单任务模型 ----------
    for name in label_names:
        model = SingleTaskModel(
            spec=spec,
            embedding_dim=cfg.model.embedding_dim,
            sparse_embeddings=cfg.model.sparse_embeddings,
            hidden=cfg.model.expert_hidden,
            dropout=cfg.model.dropout,
        )
        r = train_model(
            model, train_t, valid_t, cfg, [name], task_types, device,
            group_valid=group_valid, use_uncertainty=False, tag=f"单任务[{name}]",
        )
        results[f"single_task::{name}"] = r["metrics"]

    # ---------- 方案 B：Shared-Bottom（= 1 个专家的 MMoE）----------
    model_shared = build_mmoe(cfg, spec, num_tasks=len(label_names), num_experts=1)
    logger.info("Shared-Bottom 基线: %s", model_shared.describe())
    r_shared = train_model(
        model_shared, train_t, valid_t, cfg, label_names, task_types, device,
        group_valid=group_valid, tag="SharedBottom",
    )
    results["shared_bottom"] = r_shared["metrics"]

    return results


def _print_table(rows: List[Dict]) -> None:
    """打印对比表。"""
    header = f"{'任务':<8}{'指标':<8}{'单任务独立':>14}{'Shared-Bottom':>16}{'MMoE(多专家)':>16}{'MMoE 相对独立':>16}"
    line = "-" * len(header)
    logger.info("\n%s\n%s\n%s", line, header, line)
    for r in rows:
        better = "" if r["delta"] is None else f"{r['delta']:+.4f}"
        logger.info(
            "%-8s%-8s%14s%16s%16s%16s",
            r["task"], r["metric"], r["single"], r["shared"], r["mmoe"], better,
        )
    logger.info(line)


def run(cfg: Config, args) -> None:
    set_seed(cfg.project.seed)
    device = get_device()
    if args.epochs:
        cfg.model.epochs = args.epochs

    train_df, valid_df, spec, dense_proc = load_processed(cfg)
    train_t = build_tensors(train_df, spec, dense_proc)
    valid_t = build_tensors(valid_df, spec, dense_proc)
    label_names = list(schema.MULTI_TASK_LABELS)
    task_types = dict(schema.LABEL_TASK_TYPES)
    group_valid = valid_df[schema.InterCol.USER_ID].to_numpy()

    # ---------- 方案 C：MMoE（优先复用 train.py 已训练好的模型）----------
    ckpt = cfg.abs_path(cfg.paths.model_dir) / MODEL_FILE
    if ckpt.exists() and not args.retrain_mmoe:
        model, spec, dense_proc, bundle = load_bundle(ckpt, device)
        metrics_mmoe = evaluate(model, valid_t, label_names, task_types, device, cfg.model.batch_size, group_valid)
        logger.info("复用已训练的 MMoE 模型: %s", ckpt)
    else:
        logger.info("未找到已训练模型，现场训练 MMoE ...")
        model = build_mmoe(cfg, spec, num_tasks=len(label_names))
        r = train_model(
            model, train_t, valid_t, cfg, label_names, task_types, device,
            group_valid=group_valid, tag="MMoE",
        )
        metrics_mmoe = r["metrics"]
    logger.info("MMoE 指标: %s", {k: round(v, 4) for k, v in metrics_mmoe.items()})

    if args.skip_baselines:
        logger.info("--skip-baselines 已开启，跳过基线训练")
        return

    # ---------- 方案 A / B ----------
    baseline_results = _train_baselines(
        cfg, spec, train_t, valid_t, label_names, task_types, device, group_valid
    )

    # ---------- 汇总对比表 ----------
    rows: List[Dict] = []
    for name in label_names:
        if task_types[name] == "binary":
            metric, key = "AUC", f"{name}_auc"
        else:
            metric, key = "RMSE", f"{name}_rmse"

        single = baseline_results[f"single_task::{name}"].get(key, float("nan"))
        shared = baseline_results["shared_bottom"].get(key, float("nan"))
        mmoe = metrics_mmoe.get(key, float("nan"))

        # 二分类 AUC 越大越好，回归 RMSE 越小越好 —— 统一折算成"越大越好"的 delta
        sign = 1.0 if task_types[name] == "binary" else -1.0
        delta = sign * (mmoe - single) if not (np.isnan(mmoe) or np.isnan(single)) else None

        rows.append({
            "task": name, "metric": metric,
            "single": f"{single:.4f}", "shared": f"{shared:.4f}", "mmoe": f"{mmoe:.4f}",
            "delta": delta,
        })

    _print_table(rows)

    # ---------- 落盘 ----------
    log_dir = cfg.ensure_dir(cfg.paths.log_dir)
    flat = []
    for r in rows:
        flat.append({
            "task": r["task"], "metric": r["metric"],
            "single_task": r["single"], "shared_bottom": r["shared"], "mmoe": r["mmoe"],
            "mmoe_gain_vs_single": r["delta"],
        })
    pd.DataFrame(flat).to_csv(log_dir / "baseline_comparison.csv", index=False, encoding="utf-8")
    save_json(
        {"mmoe": metrics_mmoe, "baselines": baseline_results},
        log_dir / "baseline_comparison.json",
    )
    logger.info("对比结果已保存: %s", log_dir / "baseline_comparison.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="离线评估与基线对比")
    parser.add_argument("--config", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="覆盖 epoch 数（基线与 MMoE 一致）")
    parser.add_argument("--skip-baselines", action="store_true", help="只评估 MMoE，不训练基线")
    parser.add_argument("--retrain-mmoe", action="store_true", help="不复用已有模型，重新训练 MMoE")
    args = parser.parse_args()
    run(load_config(args.config), args)


if __name__ == "__main__":
    main()
