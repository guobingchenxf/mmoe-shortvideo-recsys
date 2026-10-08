"""多任务损失函数：固定权重 vs 不确定性加权（Uncertainty Weighting）。


"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """多任务联合损失。

    Args:
        label_names: 标签名列表，顺序必须与模型输出的顺序一致。
        task_types: {标签名: "binary" | "regression"}。
        use_uncertainty: True 使用不确定性加权；False 使用固定权重。
        manual_weights: 固定权重模式下的权重列表。
    """

    def __init__(
        self,
        label_names: Sequence[str],
        task_types: Dict[str, str],
        use_uncertainty: bool = True,
        manual_weights: Sequence[float] | None = None,
    ):
        super().__init__()
        self.label_names: List[str] = list(label_names)
        self.task_types = dict(task_types)
        self.use_uncertainty = use_uncertainty

        n = len(self.label_names)
        if manual_weights is None:
            manual_weights = [1.0] * n
        assert len(manual_weights) == n, "manual_weights 长度必须等于任务数"
        self.register_buffer("manual_weights", torch.tensor(list(manual_weights), dtype=torch.float32))

        if use_uncertainty:
            # 可学习参数 s_t = log σ_t²；初始 0 => 各任务等权起步
            self.log_vars = nn.Parameter(torch.zeros(n))

    def forward(
        self, logits: List[torch.Tensor], labels: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """计算总损失。

        Args:
            logits: 模型输出列表，每个元素形状 (B,)；二分类是 logit，回归是预测值。
            labels: {标签名: (B,)}。

        Returns:
            (总损失, {标签名: 该任务的原始损失（detach 后，便于日志观察）})
        """
        per_task: Dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=logits[0].device)

        for i, name in enumerate(self.label_names):
            pred = logits[i].reshape(-1)
            target = labels[name].reshape(-1).to(pred.dtype)

            if self.task_types[name] == "binary":
                loss_t = F.binary_cross_entropy_with_logits(pred, target)
            else:
                loss_t = F.mse_loss(pred, target)

            per_task[name] = loss_t.detach()

            if self.use_uncertainty:
                s = self.log_vars[i]
                # exp(-s)*L + 0.5*s ：不确定性越大 => 权重越小，同时用 0.5*s 防止权重塌缩到 0
                total = total + torch.exp(-s) * loss_t + 0.5 * s
            else:
                total = total + self.manual_weights[i] * loss_t

        return total, per_task

    @torch.no_grad()
    def learned_weights(self) -> Dict[str, float]:
        """返回当前各任务的实际权重 exp(-s_t)，用于观察任务间权重的分化。"""
        if not self.use_uncertainty:
            return {n: float(w) for n, w in zip(self.label_names, self.manual_weights.tolist())}
        return {
            n: float(torch.exp(-self.log_vars[i].detach()))
            for i, n in enumerate(self.label_names)
        }
