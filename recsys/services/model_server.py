"""模型服务：加载自包含模型包，提供特征编码、批量打分与预热。

为什么把「特征编码」也放在这里，而不是散落在业务代码里：
线上和线下必须用**同一份 FeatureSchema** 做编码（哈希桶数、序列长度、归一化统计量
全部来自训练时保存的 bundle）。把它收敛到 ModelServer 一个地方，
就能从机制上杜绝最难排查的一类线上事故 —— Training-Serving Skew（训练/服务特征不一致）。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from recsys.features.feature_column import FeatureSchema
from recsys.features.preprocess import DenseProcessor, encode_multi_hot, hash_encode
from recsys.models.train import load_bundle
from recsys.utils.logger import get_logger

logger = get_logger("recsys.services.model")


class ModelServer:
    """封装模型加载、特征编码与批量推理。"""

    def __init__(self, bundle_path: str | Path, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        self.model, self.spec, self.dense_proc, self.bundle = load_bundle(bundle_path, self.device)
        self.label_names: List[str] = list(self.bundle["label_names"])
        self.task_types: Dict[str, str] = dict(self.bundle["task_types"])
        self.metrics: Dict[str, float] = dict(self.bundle.get("metrics", {}))
        logger.info(
            "模型加载完成: %s | 离线指标: %s",
            self.model.describe(),
            {k: round(v, 4) for k, v in self.metrics.items()},
        )

    # ------------------------------------------------------------------
    # 特征编码（与离线 pipeline 共用同一份 spec，保证口径一致）
    # ------------------------------------------------------------------
    @property
    def dense_feature_names(self) -> List[str]:
        return self.dense_proc.feature_names

    def encode_hash(self, feature_name: str, values: List[str]) -> np.ndarray:
        """对指定特征做稳定哈希编码，桶数取自 spec。"""
        bucket = self.spec.column(feature_name).vocab_size
        return hash_encode(values, bucket)

    def encode_multi_hot(self, feature_name: str, raw_values: List[str]) -> np.ndarray:
        """多值特征编码，返回 (N, max_len) 的 id 矩阵。"""
        col = self.spec.column(feature_name)
        return encode_multi_hot(raw_values, col.vocab_size, col.max_len)

    def encode_sequence(self, feature_name: str, video_ids: List[int]) -> np.ndarray:
        """把用户行为序列编码成定长 id 向量（最新在前，不足补 0）。"""
        col = self.spec.column(feature_name)
        out = np.zeros(col.max_len, dtype=np.int64)
        if video_ids:
            ids = list(video_ids)[: col.max_len]
            out[: len(ids)] = hash_encode(ids, col.vocab_size)
        return out

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def build_batch(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """把「一条条特征字典」拼成模型需要的一批张量。"""
        dense = np.stack([self.dense_proc.transform_one(f) for f in features]).astype(np.float32)
        return {
            "dense": torch.from_numpy(dense),
            "sparse": {
                c.name: torch.tensor([int(f[c.name]) for f in features], dtype=torch.long)
                for c in self.spec.sparse
            },
            "multi_hot": {
                c.name: torch.tensor([list(f[c.name]) for f in features], dtype=torch.long)
                for c in self.spec.multi_hot
            },
            "sequence": {
                c.name: torch.tensor([list(f[c.name]) for f in features], dtype=torch.long)
                for c in self.spec.sequence
            },
            # 推理不需要标签；保留空字典是为了让 batch 结构与训练时完全一致
            "labels": {},
        }

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        moved: Dict[str, Any] = {"dense": batch["dense"].to(self.device), "labels": {}}
        for key in ("sparse", "multi_hot", "sequence"):
            moved[key] = {k: v.to(self.device) for k, v in batch[key].items()}
        return moved

    @torch.no_grad()
    def predict(self, features: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """批量打分。返回每个候选的各目标预测值（二分类为概率，回归为预测值）。"""
        if not features:
            return []
        batch = self._to_device(self.build_batch(features))
        logits = self.model(batch)

        cols: List[np.ndarray] = []
        for i, name in enumerate(self.label_names):
            out = torch.sigmoid(logits[i]) if self.task_types[name] == "binary" else logits[i]
            cols.append(out.detach().cpu().numpy())

        return [
            {name: float(cols[i][j]) for i, name in enumerate(self.label_names)}
            for j in range(len(features))
        ]

    def warmup(self, n: int = 8) -> float:
        """预热：先跑一次前向，避免第一个真实请求承担算子初始化/内存分配开销。

        线上服务发布后如果不预热，首个请求延时可能是稳态的几十倍，
        在按 P99 考核的推荐接口里是必须做的一步。
        """
        dummy: List[Dict[str, Any]] = []
        for _ in range(n):
            f: Dict[str, Any] = {name: 0.0 for name in self.dense_feature_names}
            for c in self.spec.sparse:
                f[c.name] = 1
            for c in self.spec.multi_hot + self.spec.sequence:
                f[c.name] = [0] * c.max_len
            dummy.append(f)

        t0 = time.perf_counter()
        self.predict(dummy)
        cost = (time.perf_counter() - t0) * 1000.0
        logger.info("模型预热完成: %d 条候选, 耗时 %.2fms", n, cost)
        return cost

    def info(self) -> Dict[str, Any]:
        return {
            "model": self.model.describe(),
            "labels": self.label_names,
            "task_types": self.task_types,
            "num_dense_features": len(self.spec.dense),
            "num_sparse_features": len(self.spec.sparse),
            "embedding_dim": self.bundle["model_kwargs"].get("embedding_dim"),
            "num_experts": self.bundle["model_kwargs"].get("num_experts"),
            "offline_metrics": {k: round(v, 4) for k, v in self.metrics.items()},
        }
