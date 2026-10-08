"""推荐链路业务逻辑：多路召回 -> 特征组装 -> MMoE 精排 -> 融合排序 -> 推荐理由。

=====================================================================
=====================================================================
采用漏斗式架构：
    1) 召回层：用便宜的策略从全库里快速挑出几百个"可能相关"的候选；
    2) 排序层：只对这几百个候选跑 MMoE 精排，得到各目标预测值；
    3) 融合层：把多目标预测融成一个排序分（业务上就是"综合体验"的定义）。
本项目实现了三路召回（热门 / 同类目兴趣 / 关注作者），
生产环境还会加上向量召回（双塔 + ANN）等更多路。
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from recsys.config import Config
from recsys.data import schema
from recsys.features import feature_column as fc
from recsys.services.model_server import ModelServer
from recsys.services.redis_client import CacheClient
from recsys.utils.logger import get_logger
from recsys.utils.timer import Timer

logger = get_logger("recsys.services.ranking")

# ---------------- 召回参数 ----------------
RECALL_PER_ROUTE = 200      # 每路召回的候选上限
MAX_RECENT_CATEGORIES = 3   # 用最近几个类目做兴趣召回
MAX_RECENT_AUTHORS = 3      # 用最近几个作者做关注召回
INDEX_CAP_PER_KEY = 200     # 每个类目/作者在倒排索引里保留的视频数

# ---------------- 多目标融合权重 ----------------
# 业务上"综合体验"的定义：点击最重要，点赞体现满意度，完播体现内容质量。
# 生产环境这些权重通常由线上 A/B 实验或离线拟合确定。
SCORE_WEIGHTS: Dict[str, float] = {
    "click": 0.6,
    "like": 0.3,
    "watch_time_ratio": 0.1,
}

# ---------------- 历史点击率平滑（与离线 pipeline 保持同一套参数）----------------
HIST_PRIOR = 0.05
HIST_ALPHA = 5.0

#: 冷启动用户（不在物料库中）映射到的 Embedding 槽位。
#: 生产环境更规范的做法是在词表里预留 <UNK> 槽位，并在训练时按一定概率把用户随机置为 UNK，
#: 让模型学到"未知用户"的通用表达；这里简化为落到一个合法的 0 号槽位。
UNKNOWN_USER_INDEX = 0

#: 召回来源 -> 推荐理由模板
_REASON_TEMPLATE = {
    "热门推荐": "平台近期热门内容",
    "同类目兴趣": "你最近常看「{category}」类内容",
    "关注作者": "你互动过的创作者（{author}）有新作品",
}


class Recommender:
    """在线推荐器：持有模型、物料库、召回索引、用户画像与行为缓存。"""

    def __init__(self, cfg: Config, server: ModelServer, cache: CacheClient):
        self.cfg = cfg
        self.server = server
        self.cache = cache

        self._load_catalog()
        self._build_recall_index()
        self._build_user_profiles()
        self._seed_cache()

    # ==================================================================
    # 初始化：物料库 / 召回索引 / 用户画像
    # ==================================================================
    def _load_catalog(self) -> None:
        """加载用户与视频元数据。

        为了控制单次请求的延时，这里不用 DataFrame.loc 逐条查询（那样每次约几十微秒），
        而是转成原生 dict，把查询变成 O(1) 的哈希查找。
        """
        raw_dir = self.cfg.abs_path(self.cfg.paths.data_raw)
        users = pd.read_csv(raw_dir / schema.USERS_FILE)
        videos = pd.read_csv(raw_dir / schema.VIDEOS_FILE)

        self.user_meta: Dict[int, Dict[str, Any]] = users.set_index(
            schema.UserCol.USER_ID
        ).to_dict("index")
        self.video_meta: Dict[int, Dict[str, Any]] = videos.set_index(
            schema.VideoCol.VIDEO_ID
        ).to_dict("index")
        # 保留原始 DataFrame，供构建召回倒排索引时复用（避免从 dict 反建时丢失列名）
        self._videos_df: pd.DataFrame = videos

        # 冷启动用户的默认画像：用全站中位数兜底
        self.default_user = {
            schema.UserCol.AGE: int(users[schema.UserCol.AGE].median()),
            schema.UserCol.GENDER: 2,
            schema.UserCol.CITY_LEVEL: 3,
            schema.UserCol.ACTIVE_LEVEL: 1,
            schema.UserCol.REGISTER_DAYS: 30,
        }
        logger.info("物料库加载完成: 用户=%d, 视频=%d", len(self.user_meta), len(self.video_meta))

    def _build_recall_index(self) -> None:
        """构造召回所需的倒排索引：热门榜 / 类目倒排 / 作者倒排。"""
        videos = self._videos_df.sort_values(schema.VideoCol.POPULARITY, ascending=False)

        self.popular_videos: List[int] = (
            videos[schema.VideoCol.VIDEO_ID].head(RECALL_PER_ROUTE).tolist()
        )

        self.cat_index: Dict[int, List[int]] = defaultdict(list)
        self.author_index: Dict[int, List[int]] = defaultdict(list)
        for vid, cat, author in zip(
            videos[schema.VideoCol.VIDEO_ID],
            videos[schema.VideoCol.CATEGORY],
            videos[schema.VideoCol.AUTHOR_ID],
        ):
            if len(self.cat_index[cat]) < INDEX_CAP_PER_KEY:
                self.cat_index[cat].append(int(vid))
            if len(self.author_index[author]) < INDEX_CAP_PER_KEY:
                self.author_index[author].append(int(vid))

        logger.info(
            "召回索引构建完成: 热门榜=%d, 类目数=%d, 作者数=%d",
            len(self.popular_videos), len(self.cat_index), len(self.author_index),
        )

    def _build_user_profiles(self) -> None:
        """从训练日志里还原用户画像：行为序列 + 兴趣类目/作者 + 历史点击率。

        只用训练集（train.parquet），且按时间顺序取"最近"的行为，
        与离线特征的口径保持一致；使用验证集数据会造成线上信息泄漏。
        """
        train = pd.read_parquet(
            self.cfg.abs_path(self.cfg.paths.data_processed) / "train.parquet",
            columns=[
                schema.InterCol.USER_ID, "raw_video_id", schema.InterCol.TIMESTAMP,
                schema.InterCol.CLICK, schema.VideoCol.CATEGORY, schema.VideoCol.AUTHOR_ID,
            ],
        ).sort_values(schema.InterCol.TIMESTAMP)

        max_seq = self.server.spec.column(fc.F_USER_SEQ).max_len
        seq: Dict[int, List[int]] = defaultdict(list)
        cats: Dict[int, List[int]] = defaultdict(list)
        authors: Dict[int, List[int]] = defaultdict(list)
        exposures: Dict[int, int] = defaultdict(int)
        clicks: Dict[int, int] = defaultdict(int)

        for uid, vid, clicked, cat, author in zip(
            train[schema.InterCol.USER_ID].to_numpy(),
            train["raw_video_id"].to_numpy(),
            train[schema.InterCol.CLICK].to_numpy(),
            train[schema.VideoCol.CATEGORY].to_numpy(),
            train[schema.VideoCol.AUTHOR_ID].to_numpy(),
        ):
            uid = int(uid)
            exposures[uid] += 1
            if not clicked:
                continue
            clicks[uid] += 1
            # 只保留最近的 max_seq 条点击行为作为行为序列（最新在后，稍后反转为最新在前）
            seq[uid].append(int(vid))
            if len(seq[uid]) > max_seq:
                seq[uid].pop(0)
            if int(cat) not in cats[uid]:
                cats[uid].append(int(cat))
            if int(author) not in authors[uid]:
                authors[uid].append(int(author))

        # 行为序列改为"最新在前"，与离线 pipeline 的构造方式一致
        self.user_seq: Dict[int, List[int]] = {u: list(reversed(v)) for u, v in seq.items()}
        # 兴趣类目/作者也取最近的若干个（列表前部即最近）
        self.user_cats: Dict[int, List[int]] = {
            u: list(reversed(v))[:MAX_RECENT_CATEGORIES] for u, v in cats.items()
        }
        self.user_authors: Dict[int, List[int]] = {
            u: list(reversed(v))[:MAX_RECENT_AUTHORS] for u, v in authors.items()
        }
        self.user_stats: Dict[int, Tuple[int, int]] = {
            u: (exposures[u], clicks[u]) for u in exposures
        }
        logger.info(
            "用户画像构建完成: 有行为序列的用户=%d / 总用户=%d",
            len(self.user_seq), len(exposures),
        )

    def _seed_cache(self) -> None:
        """把训练期的用户行为序列写入缓存，让服务在无外部依赖时也能立刻演示。"""
        for uid, items in self.user_seq.items():
            self.cache.set_user_sequence(uid, items, ttl=self.cfg.service.cache_ttl)
        logger.info("已向缓存写入 %d 个用户的行为序列", len(self.user_seq))

    # ==================================================================
    # 召回
    # ==================================================================
    def _recall(self, user_id: int, seq_ids: Sequence[int]) -> Dict[int, str]:
        """多路召回，返回 {video_id: 召回来源}。

        优先级：同类目兴趣 > 关注作者 > 热门。
        候选去重（同一视频只保留首次命中的来源），并过滤用户已看过的视频。
        """
        limit = self.cfg.service.recall_size
        seen = set(int(x) for x in seq_ids)
        candidates: Dict[int, str] = {}

        def add(video_ids: Sequence[int], source: str) -> bool:
            for v in video_ids:
                v = int(v)
                if v in seen or v in candidates:
                    continue
                candidates[v] = source
                if len(candidates) >= limit:
                    return True
            return False

        # 路 1：用户兴趣类目（个性化最强，放最前）
        for cat in self.user_cats.get(user_id, []):
            if add(self.cat_index.get(cat, []), "同类目兴趣"):
                return candidates
        # 路 2：用户互动过的作者
        for author in self.user_authors.get(user_id, []):
            if add(self.author_index.get(author, []), "关注作者"):
                return candidates
        # 路 3：热门兜底（保证任何用户都有候选，同时缓解冷启动）
        add(self.popular_videos, "热门推荐")
        return candidates

    # ==================================================================
    # 特征组装
    # ==================================================================
    def _default_context(self, overrides: Optional[Dict[str, Any]]) -> Dict[str, int]:
        """构造上下文特征；未传时用服务端当前时间兜底。"""
        now = datetime.now()
        ctx = {"hour": now.hour, "dow": now.weekday(), "device": 1, "source": 0}
        overrides = overrides or {}
        for key in ("hour", "dow", "device", "source"):
            if overrides.get(key) is not None:
                ctx[key] = int(overrides[key])

        # 防御性裁剪：外部传入的上下文可能越界（例如 device=99），
        # 直接拿去查 Embedding 会 IndexError，因此在边界处统一取模兜底。
        # 这正是"只在系统边界做校验"的典型场景——内部数据已保证合法。
        ctx["hour"] %= 24
        ctx["dow"] %= 7
        ctx["device"] %= self.server.spec.column(fc.F_DEVICE).vocab_size
        ctx["source"] %= self.server.spec.column(fc.F_SOURCE).vocab_size
        return ctx

    def _make_features(
        self,
        user_id: int,
        video_ids: List[int],
        ctx: Dict[str, int],
        seq_ids: Sequence[int],
    ) -> List[Dict[str, Any]]:
        """为一批候选视频组装模型输入特征。

        编码尽量批量做（一次处理全部候选），避免逐条调用带来的 Python 开销 ——
        实测批量编码比逐条快一个数量级。
        """
        # 冷启动用户没有画像，用全站中位数兜底，并把 id 映射到合法的 Embedding 槽位
        known_user = user_id in self.user_meta
        um = self.user_meta[user_id] if known_user else self.default_user
        user_index = user_id if known_user else UNKNOWN_USER_INDEX
        expo, clk = self.user_stats.get(user_id, (0, 0))
        # 与离线 pipeline 相同的贝叶斯平滑口径
        hist_rate = (clk + HIST_ALPHA * HIST_PRIOR) / (expo + HIST_ALPHA)
        hist_cnt = float(expo)

        hour = ctx["hour"]
        dow = ctx["dow"]
        hour_sin, hour_cos = math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24)
        dow_sin, dow_cos = math.sin(2 * math.pi * dow / 7), math.cos(2 * math.pi * dow / 7)

        metas = [self.video_meta[v] for v in video_ids]
        cat_ids = [int(m[schema.VideoCol.CATEGORY]) for m in metas]

        # ---- 批量编码（与离线共用同一份 spec）----
        video_hash = self.server.encode_hash(fc.F_VIDEO_ID, [str(v) for v in video_ids])
        cross_hash = self.server.encode_hash(
            fc.F_USER_CAT_CROSS, [f"{user_id}_{c}" for c in cat_ids]
        )
        genres_mat = self.server.encode_multi_hot(
            fc.F_GENRES, [str(m[schema.VideoCol.GENRES]) for m in metas]
        )
        seq_vec = self.server.encode_sequence(fc.F_USER_SEQ, list(seq_ids))

        features: List[Dict[str, Any]] = []
        for i, meta in enumerate(metas):
            features.append({
                # ---- dense：原始值，由 DenseProcessor 统一做 log1p + 标准化 ----
                fc.F_AGE: um[schema.UserCol.AGE],
                fc.F_DURATION: meta[schema.VideoCol.DURATION_MS],
                fc.F_POPULARITY: meta[schema.VideoCol.POPULARITY],
                fc.F_UPLOAD_DAYS: meta[schema.VideoCol.UPLOAD_DAYS_AGO],
                fc.F_REGISTER_DAYS: um[schema.UserCol.REGISTER_DAYS],
                fc.F_ACTIVE_LEVEL: um[schema.UserCol.ACTIVE_LEVEL],
                fc.F_CITY_LEVEL: um[schema.UserCol.CITY_LEVEL],
                fc.F_HOUR_SIN: hour_sin,
                fc.F_HOUR_COS: hour_cos,
                fc.F_DOW_SIN: dow_sin,
                fc.F_DOW_COS: dow_cos,
                fc.F_USER_HIST_CTR: hist_rate,
                fc.F_USER_HIST_CNT: hist_cnt,
                # ---- sparse ----
                fc.F_USER_ID: user_index,
                fc.F_VIDEO_ID: int(video_hash[i]),
                fc.F_AUTHOR_ID: int(meta[schema.VideoCol.AUTHOR_ID]),
                fc.F_CATEGORY: cat_ids[i],
                fc.F_GENDER: um[schema.UserCol.GENDER],
                fc.F_DEVICE: ctx["device"],
                fc.F_SOURCE: ctx["source"],
                fc.F_USER_CAT_CROSS: int(cross_hash[i]),
                # ---- multi_hot / sequence ----
                fc.F_GENRES: [int(x) for x in genres_mat[i]],
                fc.F_USER_SEQ: [int(x) for x in seq_vec],
            })
        return features

    # ==================================================================
    # 主入口
    # ==================================================================
    def recommend(
        self,
        user_id: int,
        context_features: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """一次完整的推荐请求。"""
        timer = Timer()
        is_cold_start = user_id not in self.user_meta

        ctx = self._default_context(context_features)
        seq_ids = self.cache.get_user_sequence(user_id)
        timer.mark("context")

        candidates = self._recall(user_id, seq_ids)
        timer.mark("recall")

        video_ids = list(candidates.keys())
        features = self._make_features(user_id, video_ids, ctx, seq_ids)
        timer.mark("features")

        preds = self.server.predict(features)
        timer.mark("inference")

        items: List[Dict[str, Any]] = []
        for vid, pred in zip(video_ids, preds):
            # 多目标融合：把三个目标的预测值按业务权重加权成一个排序分
            fused = sum(SCORE_WEIGHTS.get(name, 0.0) * pred.get(name, 0.0) for name in pred)
            meta = self.video_meta[vid]
            source = candidates[vid]
            reason = _REASON_TEMPLATE.get(source, "为你推荐").format(
                category=meta[schema.VideoCol.CATEGORY],
                author=meta[schema.VideoCol.AUTHOR_ID],
            )
            items.append({
                "video_id": int(vid),
                "score": round(float(fused), 6),
                "reason": reason,
                "recall_source": source,
                "task_scores": {k: round(float(v), 6) for k, v in pred.items()},
            })

        items.sort(key=lambda x: x["score"], reverse=True)
        timer.mark("rank")

        return {
            "user_id": user_id,
            "is_cold_start": is_cold_start,
            "video_list": items[:top_k],
            "candidate_count": len(video_ids),
            "latency_ms": round(timer.elapsed_ms, 3),
            "latency_breakdown": timer.breakdown(),
        }

    # ==================================================================
    def info(self) -> Dict[str, Any]:
        return {
            "catalog_size": {"users": len(self.user_meta), "videos": len(self.video_meta)},
            "cache_backend": self.cache.backend_name,
            "recall_routes": ["同类目兴趣", "关注作者", "热门推荐"],
            "score_weights": SCORE_WEIGHTS,
        }

    def close(self) -> None:
        self.cache.close()
