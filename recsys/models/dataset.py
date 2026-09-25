"""把特征 parquet 转换成 PyTorch 张量，并封装为 Dataset / DataLoader。

效率设计：不在 __getitem__ 里逐样本做 Python 级别的字典拼装（那样 10 万样本会慢一个量级），
而是让 Dataset 只返回行号，真正的批量拼装交给 collate_fn 在张量层面用花式索引一次完成。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from recsys.data import schema
from recsys.features.feature_column import FeatureSchema
from recsys.features.preprocess import DenseProcessor


@dataclass
class RecTensors:
    """一份数据集对应的全部特征张量（全部驻留内存，本项目规模下完全可行）。"""

    dense: torch.Tensor                          # (N, D) float32
    sparse: Dict[str, torch.Tensor]              # name -> (N,) int64
    multi_hot: Dict[str, torch.Tensor]           # name -> (N, L) int64
    sequence: Dict[str, torch.Tensor]            # name -> (N, L) int64
    labels: Dict[str, torch.Tensor]              # name -> (N,) float32

    @property
    def n(self) -> int:
        return int(self.dense.shape[0])

    def gather(self, idx: torch.Tensor) -> Dict[str, Any]:
        """按行号取一个 batch。"""
        return {
            "dense": self.dense[idx],
            "sparse": {k: v[idx] for k, v in self.sparse.items()},
            "multi_hot": {k: v[idx] for k, v in self.multi_hot.items()},
            "sequence": {k: v[idx] for k, v in self.sequence.items()},
            "labels": {k: v[idx] for k, v in self.labels.items()},
        }

    def subset(self, indices: np.ndarray) -> "RecTensors":
        """取子集（用于快速调试 / 抽样训练）。"""
        idx = torch.as_tensor(indices, dtype=torch.long)
        return RecTensors(
            dense=self.dense[idx],
            sparse={k: v[idx] for k, v in self.sparse.items()},
            multi_hot={k: v[idx] for k, v in self.multi_hot.items()},
            sequence={k: v[idx] for k, v in self.sequence.items()},
            labels={k: v[idx] for k, v in self.labels.items()},
        )

    def to(self, device: torch.device) -> "RecTensors":
        return RecTensors(
            dense=self.dense.to(device),
            sparse={k: v.to(device) for k, v in self.sparse.items()},
            multi_hot={k: v.to(device) for k, v in self.multi_hot.items()},
            sequence={k: v.to(device) for k, v in self.sequence.items()},
            labels={k: v.to(device) for k, v in self.labels.items()},
        )


def _stack_list_column(df: pd.DataFrame, name: str, max_len: int) -> np.ndarray:
    """把 parquet 中的 list 列整理成 (N, max_len) 的 int64 矩阵。"""
    mat = np.zeros((len(df), max_len), dtype=np.int64)
    for i, row in enumerate(df[name].to_numpy()):
        arr = np.asarray(row, dtype=np.int64).ravel()[:max_len]
        mat[i, : len(arr)] = arr
    return mat


def build_tensors(df: pd.DataFrame, spec: FeatureSchema, dense_proc: DenseProcessor) -> RecTensors:
    """DataFrame -> RecTensors。dense 特征在此处完成标准化。"""
    dense = torch.from_numpy(dense_proc.transform(df).astype(np.float32, copy=False))

    sparse = {
        c.name: torch.from_numpy(df[c.name].to_numpy(dtype=np.int64, copy=False))
        for c in spec.sparse
    }
    multi_hot = {
        c.name: torch.from_numpy(_stack_list_column(df, c.name, c.max_len))
        for c in spec.multi_hot
    }
    sequence = {
        c.name: torch.from_numpy(_stack_list_column(df, c.name, c.max_len))
        for c in spec.sequence
    }
    labels = {
        name: torch.from_numpy(df[name].to_numpy(dtype=np.float32, copy=False))
        for name in schema.MULTI_TASK_LABELS
    }
    return RecTensors(dense=dense, sparse=sparse, multi_hot=multi_hot, sequence=sequence, labels=labels)


class RecDataset(Dataset):
    """只返回行号，真正的拼装交给 collate_fn。"""

    def __init__(self, data: RecTensors):
        self.data = data

    def __len__(self) -> int:
        return self.data.n

    def __getitem__(self, i: int) -> int:
        return i


def collate_indices(indices: List[int], data: RecTensors) -> Dict[str, Any]:
    """把一批行号拼成一个 batch 的张量字典。"""
    idx = torch.as_tensor(indices, dtype=torch.long)
    return data.gather(idx)


def make_loader(
    data: RecTensors,
    batch_size: int,
    shuffle: bool,
    drop_last: bool = False,
) -> DataLoader:
    """构造 DataLoader（Windows 下 num_workers 必须为 0，否则多进程会报错）。"""
    return DataLoader(
        RecDataset(data),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        collate_fn=partial(collate_indices, data=data),
    )
