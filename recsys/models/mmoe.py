"""MMoE：Multi-gate Mixture-of-Experts 多目标模型。

"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from recsys.features.feature_column import PAD_INDEX, FeatureSchema


def _mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    dropout: float = 0.0,
) -> nn.Sequential:
    """构造一个 MLP：hidden 层后接 ReLU + Dropout，最后一层为纯线性。"""
    layers: List[nn.Module] = []
    d = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class EmbeddingInputLayer(nn.Module):
    """特征输入层：把 dense / sparse / multi_hot / sequence 编码成一个稠密向量。

    这一层被 MMoE 与单任务基线复用，保证二者**输入特征完全一致**，
    从而让"MMoE vs 单任务"的对比是公平的。
    """

    def __init__(self, spec: FeatureSchema, embedding_dim: int = 16, sparse_embeddings: bool = True):
        """
        Args:
            sparse_embeddings: 是否让 Embedding 产生**稀疏梯度**。

            为什么默认开启：本项目有 4 张 10 万级词表的 Embedding（video_id / 交叉特征 /
            genres / 行为序列），总参数量 6.5M+。若用稠密梯度，每个 step 都要为整张表
            计算梯度并用 Adam 更新所有行（实测优化器单步 0.29s，是训练最慢的一环）。
            开启稀疏梯度后只更新命中的行，配合 SparseAdam 可显著提速——
            这正是工业界处理"超大规模稀疏特征"的标准做法。
        """
        super().__init__()
        self.spec = spec
        self.embedding_dim = embedding_dim
        self.sparse_embeddings = sparse_embeddings

        # 单值类别：直接查表
        self.sparse_emb = nn.ModuleDict(
            {c.name: nn.Embedding(c.vocab_size, embedding_dim, sparse=sparse_embeddings)
             for c in spec.sparse}
        )
        # 多值 / 序列：需要 padding，padding_idx=0 保证 PAD 位置不参与梯度
        self.multi_hot_emb = nn.ModuleDict(
            {c.name: nn.Embedding(c.vocab_size, embedding_dim, padding_idx=PAD_INDEX, sparse=sparse_embeddings)
             for c in spec.multi_hot}
        )
        self.sequence_emb = nn.ModuleDict(
            {c.name: nn.Embedding(c.vocab_size, embedding_dim, padding_idx=PAD_INDEX, sparse=sparse_embeddings)
             for c in spec.sequence}
        )

        n_emb_features = len(spec.sparse) + len(spec.multi_hot) + len(spec.sequence)
        self.output_dim = len(spec.dense) + embedding_dim * n_emb_features

    def sparse_parameters(self) -> List[nn.Parameter]:
        """返回所有 Embedding 权重。

        这些参数在 sparse=True 时产生的是稀疏梯度，必须交给 SparseAdam 更新，
        不能与稠密参数混在同一个 Adam 里（PyTorch 的 Adam 不支持稀疏梯度）。
        """
        modules = [
            *self.sparse_emb.values(),
            *self.multi_hot_emb.values(),
            *self.sequence_emb.values(),
        ]
        return [m.weight for m in modules]

    @staticmethod
    def _masked_mean(emb: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """对 (B, L, D) 做 masked mean，忽略 padding（id == 0）的位置。"""
        mask = (ids != PAD_INDEX).float().unsqueeze(-1)        # (B, L, 1)
        summed = (emb * mask).sum(dim=1)                       # (B, D)
        count = mask.sum(dim=1).clamp(min=1.0)                 # (B, 1) 防止除零
        return summed / count

    def forward(self, batch: Dict) -> torch.Tensor:
        """batch -> (B, output_dim) 的稠密向量。"""
        parts: List[torch.Tensor] = [batch["dense"]]

        for name, ids in batch["sparse"].items():
            parts.append(self.sparse_emb[name](ids))

        for name, ids in batch["multi_hot"].items():
            parts.append(self._masked_mean(self.multi_hot_emb[name](ids), ids))

        for name, ids in batch["sequence"].items():
            parts.append(self._masked_mean(self.sequence_emb[name](ids), ids))

        return torch.cat(parts, dim=-1)


class Gate(nn.Module):
    """门控网络：输出 N 个专家的权重（softmax 归一化）。

    原论文使用"单层线性 + softmax"；这里额外支持 MLP 形式的门控
    （config 中 gate_hidden 非空即启用），可用于增强门控的表达能力。
    """

    def __init__(self, in_dim: int, num_experts: int, hidden: Sequence[int] = (), dropout: float = 0.0):
        super().__init__()
        self.net = _mlp(in_dim, hidden, num_experts, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)


class MMoE(nn.Module):
    """多门控混合专家多任务模型。

    Args:
        spec: 特征声明（决定 Embedding 表规模）。
        num_tasks: 任务数量（= 塔的个数 = 门控的个数）。
        embedding_dim: 稀疏特征 Embedding 维度。
        shared_bottom_hidden: 共享底层结构；为空列表则退化为原论文结构。
        num_experts: 专家个数。
        expert_hidden: 专家网络结构。
        gate_hidden: 门控网络结构（空 = 线性门控）。
        tower_hidden: 任务塔结构。
        dropout: Dropout 比例。
    """

    def __init__(
        self,
        spec: FeatureSchema,
        num_tasks: int,
        embedding_dim: int = 16,
        sparse_embeddings: bool = True,
        shared_bottom_hidden: Sequence[int] = (256,),
        num_experts: int = 8,
        expert_hidden: Sequence[int] = (128, 64),
        gate_hidden: Sequence[int] = (),
        tower_hidden: Sequence[int] = (64, 32),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_experts = num_experts

        # ---------- 1. 输入层 ----------
        self.input_layer = EmbeddingInputLayer(spec, embedding_dim, sparse_embeddings)
        in_dim = self.input_layer.output_dim

        # ---------- 2. 共享底层（可选）----------
        if shared_bottom_hidden:
            hidden = list(shared_bottom_hidden)
            self.shared_bottom: nn.Module = _mlp(in_dim, hidden[:-1], hidden[-1], dropout)
            expert_in_dim = hidden[-1]
        else:
            self.shared_bottom = nn.Identity()
            expert_in_dim = in_dim

        # ---------- 3. 专家层：N 个专家，输入相同、参数独立 ----------
        expert_out_dim = list(expert_hidden)[-1]
        self.experts = nn.ModuleList(
            [_mlp(expert_in_dim, list(expert_hidden)[:-1], expert_out_dim, dropout)
             for _ in range(num_experts)]
        )

        # ---------- 4. 门控层：每个任务一个 ----------
        # 门控的输入与专家保持一致（都作用于共享表示），这是原论文的做法
        self.gates = nn.ModuleList(
            [Gate(expert_in_dim, num_experts, gate_hidden, dropout) for _ in range(num_tasks)]
        )

        # ---------- 5. 塔层：每个任务一个 ----------
        self.towers = nn.ModuleList(
            [_mlp(expert_out_dim, list(tower_hidden)[:-1], list(tower_hidden)[-1], dropout)
             for _ in range(num_tasks)]
        )
        self.task_heads = nn.ModuleList([nn.Linear(list(tower_hidden)[-1], 1) for _ in range(num_tasks)])

    def forward(self, batch: Dict) -> List[torch.Tensor]:
        """返回长度为 num_tasks 的列表，每个元素形状 (B,)：二分类为 logit，回归为预测值。"""
        x = self.input_layer(batch)
        h = self.shared_bottom(x)

        # (B, E, H)：所有专家的输出堆叠
        expert_outs = torch.stack([expert(h) for expert in self.experts], dim=1)

        outputs: List[torch.Tensor] = []
        for gate, tower, head in zip(self.gates, self.towers, self.task_heads):
            g = gate(h)                                                    # (B, E)
            # 按门控权重对各专家输出加权求和 —— MMoE 的关键一步
            fused = torch.bmm(g.unsqueeze(1), expert_outs).squeeze(1)      # (B, H)
            outputs.append(head(tower(fused)).squeeze(-1))                 # (B,)
        return outputs

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def sparse_parameters(self) -> List[nn.Parameter]:
        """需要 SparseAdam 更新的 Embedding 参数。"""
        return self.input_layer.sparse_parameters()

    def dense_parameters(self) -> List[nn.Parameter]:
        """其余稠密参数（共享底层 / 专家 / 门控 / 塔），交给普通 Adam 更新。"""
        sparse_ids = {id(p) for p in self.sparse_parameters()}
        return [p for p in self.parameters() if id(p) not in sparse_ids]

    def describe(self) -> str:
        return (
            f"MMoE(experts={self.num_experts}, tasks={self.num_tasks}, "
            f"input_dim={self.input_layer.output_dim}, "
            f"sparse_emb={self.input_layer.sparse_embeddings}, "
            f"params={self.num_parameters():,})"
        )


class SingleTaskModel(nn.Module):
    """单任务基线模型：共用同一套输入层结构，但没有专家与门控，直接一个塔输出。

    用于和 MMoE 做对比，回答"多任务共享到底有没有用"这个问题。

    注意：forward 返回的是**长度为 1 的列表**，以保持与 MMoE 相同的接口，
    从而复用同一套训练/评估代码。
    """

    def __init__(
        self,
        spec: FeatureSchema,
        embedding_dim: int = 16,
        sparse_embeddings: bool = True,
        hidden: Sequence[int] = (128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_layer = EmbeddingInputLayer(spec, embedding_dim, sparse_embeddings)
        hidden = list(hidden)
        self.tower = _mlp(self.input_layer.output_dim, hidden[:-1], hidden[-1], dropout)
        self.head = nn.Linear(hidden[-1], 1)

    def forward(self, batch: Dict) -> List[torch.Tensor]:
        x = self.input_layer(batch)
        return [self.head(self.tower(x)).squeeze(-1)]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def sparse_parameters(self) -> List[nn.Parameter]:
        return self.input_layer.sparse_parameters()

    def dense_parameters(self) -> List[nn.Parameter]:
        sparse_ids = {id(p) for p in self.sparse_parameters()}
        return [p for p in self.parameters() if id(p) not in sparse_ids]
