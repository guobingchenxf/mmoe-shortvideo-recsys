"""离线/在线共用的一批特征变换算子（Step 2）。

包含：
    hash_encode            稳定哈希编码
    DenseProcessor         强偏态 log1p + z-score 标准化（统计量仅由训练集拟合）
    encode_multi_hot       多值特征 -> 定长 id 序列（0 表示 PAD）
    build_user_sequence    构造"用户历史行为序列"（严格只用过去，防泄漏）
    build_user_history_stats 构造"截至当前的历史点击率/曝光数"（防泄漏）


"""

from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from recsys.data import schema
from recsys.features.feature_column import ColumnType, FeatureColumn

#: 哈希种子，改它会整体改变特征索引空间，必须离线/在线保持一致
HASH_SEED = 0


# =====================================================================
# 哈希编码
# =====================================================================
def _hash_one(value: str, bucket_size: int, seed: int = HASH_SEED) -> int:
    """把字符串稳定地映射到 [1, bucket_size-1]（0 号位保留给 PAD/UNK）。"""
    if bucket_size <= 1:
        raise ValueError(f"bucket_size 必须 > 1，收到 {bucket_size}")
    digest = hashlib.blake2b(f"{seed}:{value}".encode("utf-8"), digest_size=8).digest()
    return 1 + int.from_bytes(digest, "big") % (bucket_size - 1)


def hash_encode(
    values: Iterable,
    bucket_size: int,
    seed: int = HASH_SEED,
    cache: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    """对一批取值做稳定哈希编码，返回 int64 数组。

    Args:
        values: 待编码的取值序列（会被转成 str 后哈希）。
        bucket_size: 哈希桶数（= Embedding 的 num_embeddings）。
        seed: 哈希种子。
        cache: 可选的取值->桶号缓存。ID 类特征重复率高，缓存能显著加速，
               同时保证同一取值在整个流程中映射一致。

    Returns:
        np.ndarray[int64]，取值区间 [1, bucket_size-1]。
    """
    cache = {} if cache is None else cache
    out = np.empty(len(values), dtype=np.int64)  # type: ignore[arg-type]
    for i, v in enumerate(values):
        key = v if isinstance(v, str) else str(v)
        idx = cache.get(key, -1)
        if idx < 0:
            idx = _hash_one(key, bucket_size, seed)
            cache[key] = idx
        out[i] = idx
    return out


# =====================================================================
# Dense 标准化
# =====================================================================
class DenseProcessor:
    """连续特征处理器：可选 log1p 去偏态 + z-score 标准化。

    为什么统计量必须只在训练集上 fit：
    如果用「全量数据（含验证集）」算 mean/std，验证集的信息就泄漏进了训练过程，
    离线指标会虚高、上线后掉点。这是特征工程里最常见的一类泄漏。
    """

    def __init__(self, columns: List[FeatureColumn]):
        self.columns: List[FeatureColumn] = [c for c in columns if c.ctype == ColumnType.DENSE]
        self.mean_: Dict[str, float] = {}
        self.std_: Dict[str, float] = {}
        self._fitted = False

    @property
    def feature_names(self) -> List[str]:
        return [c.name for c in self.columns]

    def _raw(self, df: pd.DataFrame, col: FeatureColumn) -> np.ndarray:
        """取出原始值并按需做 log1p（不改动原 DataFrame）。"""
        x = pd.to_numeric(df[col.name], errors="coerce").to_numpy(dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if col.log1p:
            x = np.log1p(np.maximum(x, 0.0))
        return x

    def fit(self, df: pd.DataFrame) -> "DenseProcessor":
        """仅用训练集拟合 mean / std。"""
        for c in self.columns:
            x = self._raw(df, c)
            self.mean_[c.name] = float(x.mean())
            std = float(x.std())
            self.std_[c.name] = std if std > 1e-8 else 1.0
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """批量变换，返回 (N, num_dense) 的 float32 矩阵。"""
        if not self._fitted:
            raise RuntimeError("DenseProcessor 尚未 fit，请先调用 fit()")
        out = np.empty((len(df), len(self.columns)), dtype=np.float32)
        for j, c in enumerate(self.columns):
            x = self._raw(df, c)
            out[:, j] = (x - self.mean_[c.name]) / self.std_[c.name]
        return out

    def transform_one(self, features: Dict[str, float]) -> np.ndarray:
        """在线服务用：对单条样本的原始特征字典做同样的变换。

        与 transform 共用同一份 mean/std 和 log1p 逻辑，从根本上保证
        「离线训练」与「线上推理」的特征口径一致。
        """
        if not self._fitted:
            raise RuntimeError("DenseProcessor 尚未 fit，请先调用 fit()")
        out = np.empty(len(self.columns), dtype=np.float32)
        for j, c in enumerate(self.columns):
            x = float(features.get(c.name, 0.0) or 0.0)
            if c.log1p:
                x = float(np.log1p(max(x, 0.0)))
            out[j] = (x - self.mean_[c.name]) / self.std_[c.name]
        return out

    # ---------- 持久化 ----------
    def state_dict(self) -> Dict:
        return {"mean": self.mean_, "std": self.std_}

    def load_state_dict(self, state: Dict) -> "DenseProcessor":
        self.mean_ = {k: float(v) for k, v in state["mean"].items()}
        self.std_ = {k: float(v) for k, v in state["std"].items()}
        self._fitted = True
        return self


# =====================================================================
# Multi-hot 编码
# =====================================================================
def encode_multi_hot(
    values: Iterable,
    bucket_size: int,
    max_len: int,
    sep: str = schema.GENRE_SEP,
    seed: int = HASH_SEED,
    cache: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    """把 "a|b|c" 形式的多值字段编码成 (N, max_len) 的 id 矩阵。

    超出 max_len 的部分截断，不足的部分用 0（PAD）填充。
    """
    cache = {} if cache is None else cache
    out = np.zeros((len(values), max_len), dtype=np.int64)  # type: ignore[arg-type]
    for i, raw in enumerate(values):
        if raw is None:
            continue
        s = str(raw)
        if not s:
            continue
        ids: List[int] = []
        for tok in s.split(sep):
            tok = tok.strip()
            if not tok:
                continue
            idx = cache.get(tok)
            if idx is None:
                idx = _hash_one(tok, bucket_size, seed)
                cache[tok] = idx
            ids.append(idx)
            if len(ids) >= max_len:
                break
        if ids:
            out[i, : len(ids)] = ids
    return out


# =====================================================================
# 行为序列 & 历史统计（严格防泄漏）
# =====================================================================
def build_user_sequence(
    interactions: pd.DataFrame,
    user_col: str,
    item_col: str,
    label_col: str,
    time_col: str,
    max_len: int,
    bucket_size: int,
    seed: int = HASH_SEED,
) -> np.ndarray:
    """构造「该用户在此时刻之前点击过的视频序列」（最新在最左，0 填充）。

    实现要点（防泄漏）：
      先按时间升序排列，再逐个用户维护一个 maxlen 的历史队列；
      **先记录当前样本的序列，再把当前样本（若为点击）入队**。
      这样当前样本永远不会看到自己或未来，完全等价于线上"用历史预测当前"。
    """
    n = len(interactions)
    seq = np.zeros((n, max_len), dtype=np.int64)

    time_arr = pd.to_datetime(interactions[time_col]).to_numpy()
    order = np.argsort(time_arr, kind="stable")  # 稳定排序：同一时刻保持原顺序

    users = interactions[user_col].to_numpy()
    items = interactions[item_col].to_numpy()
    labels = interactions[label_col].to_numpy()

    # 先把所有出现过的视频一次性哈希好，循环里直接查表
    cache: Dict[str, int] = {}
    history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=max_len))

    for i in order:
        h = history[users[i]]
        if h:
            # deque 内部是「旧 -> 新」，反转得到「新 -> 旧」，左对齐写入，右侧补 0
            recent = list(h)[::-1]
            seq[i, : len(recent)] = recent
        if labels[i] == 1:
            key = str(items[i])
            idx = cache.get(key)
            if idx is None:
                idx = _hash_one(key, bucket_size, seed)
                cache[key] = idx
            h.append(idx)
    return seq


def build_user_history_stats(
    interactions: pd.DataFrame,
    user_col: str,
    label_col: str,
    time_col: str,
    prior: float = 0.05,
    alpha: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """构造「截至当前样本之前」的用户历史点击率与曝光次数。

    同样严格防泄漏：先记录、再更新计数。

    平滑：rate = (历史点击 + alpha*prior) / (历史曝光 + alpha)。
    新用户曝光少时结果向全局先验 prior 收缩，避免 0/0 和极端值。

    Returns:
        (hist_click_rate, hist_exposure_cnt) 两个 float32 数组。
    """
    n = len(interactions)
    hist_rate = np.zeros(n, dtype=np.float32)
    hist_cnt = np.zeros(n, dtype=np.float32)

    time_arr = pd.to_datetime(interactions[time_col]).to_numpy()
    order = np.argsort(time_arr, kind="stable")

    users = interactions[user_col].to_numpy()
    labels = interactions[label_col].to_numpy()

    exposures: Dict[int, int] = defaultdict(int)
    clicks: Dict[int, int] = defaultdict(int)

    for i in order:
        u = users[i]
        c = exposures[u]
        k = clicks[u]
        # 用「进入本样本之前」的累计量计算，天然不含当前样本，无泄漏
        hist_rate[i] = (k + alpha * prior) / (c + alpha)
        hist_cnt[i] = c
        exposures[u] = c + 1
        clicks[u] = k + int(labels[i])

    return hist_rate, hist_cnt
