"""FastAPI 在线推理服务入口。

接口：
    POST /recommend   推荐接口
    GET  /health      健康检查（含模型信息、缓存后端、延时统计）
    GET  /            服务信息
    GET  /docs        自动生成的接口文档（FastAPI 自带）

启动：
    python -m recsys.services.main
    # 或
    uvicorn recsys.services.main:app --host 0.0.0.0 --port 8000

环境变量（见 .env.example）：
    MODEL_PATH   模型权重路径（默认 artifacts/models/best_mmoe.pt）
    REDIS_URL    Redis 地址；留空则使用进程内内存缓存
    API_HOST / API_PORT / LOG_LEVEL
    TORCH_NUM_THREADS  推理线程数（默认 4，避免与多 worker 争抢 CPU）
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from recsys.config import Config, load_config
from recsys.models.train import MODEL_FILE
from recsys.services.model_server import ModelServer
from recsys.services.ranking import Recommender
from recsys.services.redis_client import build_cache
from recsys.services.schemas import HealthResponse, RecommendRequest, RecommendResponse
from recsys.utils.logger import get_logger
from recsys.utils.timer import LatencyTracker

logger = get_logger("recsys.services.api")

#: 全局延时统计器（滑动窗口），用于压测与线上监控
LATENCY = LatencyTracker(window=2000)


def build_recommender(cfg: Config) -> Recommender:
    """组装推荐器：模型 -> 缓存 -> 业务逻辑。"""
    # 限制推理线程数。
    # 实测（见 README 的压测数据）：本机 PyTorch 在 8 逻辑核上开 4/8 线程时，
    # 小批量推理会出现 OpenMP 线程超订，P99 从几十毫秒劣化到 1.4s。
    # 推荐接口是小批量、低延时的负载，"少线程" 反而更稳更快，故默认 1。
    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))

    bundle_path = Path(os.getenv("MODEL_PATH") or (cfg.abs_path(cfg.paths.model_dir) / MODEL_FILE))
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"未找到模型文件 {bundle_path}，请先运行: python -m recsys.models.train"
        )

    server = ModelServer(bundle_path)
    # 预热时用「真实召回规模」的批量，确保首个真实请求的张量形状也已被算子初始化覆盖
    server.warmup(cfg.service.recall_size)

    redis_url = os.getenv("REDIS_URL") or cfg.service.redis_url
    cache = build_cache(redis_url, cfg.service.cache_ttl)
    return Recommender(cfg, server, cache)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时加载一次模型与物料，进程内常驻复用。"""
    cfg = load_config()
    app.state.cfg = cfg
    app.state.recommender = build_recommender(cfg)
    logger.info("服务启动完成，接口文档: http://127.0.0.1:%s/docs", os.getenv("API_PORT", cfg.service.port))
    yield
    app.state.recommender.close()
    logger.info("服务已关闭")


app = FastAPI(
    title="MMoE 短视频推荐引擎",
    description="基于 MMoE 多目标模型的实时短视频推荐服务（召回 -> 精排 -> 重排）",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", summary="服务信息")
def root() -> dict:
    return {
        "service": "mmoe-shortvideo-recsys",
        "version": "0.1.0",
        "endpoints": ["POST /recommend", "GET /health", "GET /docs"],
    }


@app.get("/health", response_model=HealthResponse, summary="健康检查")
def health() -> HealthResponse:
    rec: Recommender = app.state.recommender
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_info=rec.server.info(),
        cache_backend=rec.cache.backend_name,
        catalog_size=rec.info()["catalog_size"],
        latency_stats=LATENCY.stats(),
    )


@app.post("/recommend", response_model=RecommendResponse, summary="获取推荐列表")
def recommend(req: RecommendRequest) -> RecommendResponse:
    """推荐主接口。

    流程：读取用户行为序列 -> 多路召回 -> 组装特征 -> MMoE 批量打分 -> 融合排序。
    """
    rec: Recommender = app.state.recommender
    try:
        result = rec.recommend(req.user_id, req.context_features, req.top_k)
    except Exception as e:  # 兜底：单次请求失败不应打挂整个服务
        logger.exception("推荐失败 user_id=%s", req.user_id)
        raise HTTPException(status_code=500, detail=f"推荐失败: {e}") from e

    LATENCY.record(result["latency_ms"])
    return RecommendResponse(**result)


def main() -> None:
    cfg = load_config()
    host = os.getenv("API_HOST", cfg.service.host)
    port = int(os.getenv("API_PORT", cfg.service.port))
    logger.info("启动服务 %s:%s", host, port)
    uvicorn.run(
        "recsys.services.main:app",
        host=host,
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
