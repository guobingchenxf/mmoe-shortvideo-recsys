"""在线服务的请求/响应契约（Pydantic 模型）。

"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    """推荐请求。"""

    user_id: int = Field(..., description="用户 id", examples=[123])
    top_k: int = Field(10, ge=1, le=100, description="返回条数")
    context_features: Optional[Dict[str, Any]] = Field(
        default=None,
        description="上下文特征，可覆盖 hour / device / source 等；缺省时服务端自动取当前时间",
        examples=[{"hour": 20, "device": 1, "source": 2}],
    )


class RecommendItem(BaseModel):
    """单条推荐结果。"""

    video_id: int = Field(..., description="视频 id")
    score: float = Field(..., description="融合后的排序分")
    reason: str = Field(..., description="推荐理由（可解释性）")
    recall_source: str = Field(..., description="该候选来自哪一路召回")
    task_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="各目标预测值：pctr / plike / 预测完播率",
    )


class RecommendResponse(BaseModel):
    """推荐响应。"""

    user_id: int = Field(..., description="用户 id")
    is_cold_start: bool = Field(False, description="是否为冷启动用户（无历史画像/行为）")
    video_list: List[RecommendItem] = Field(default_factory=list, description="推荐列表")
    candidate_count: int = Field(0, description="进入精排的候选数")
    latency_ms: float = Field(..., description="服务端总耗时（毫秒）")
    latency_breakdown: Dict[str, float] = Field(
        default_factory=dict, description="分段耗时：召回 / 组装特征 / 模型推理 / 排序"
    )


class HealthResponse(BaseModel):
    """健康检查与运行时信息。"""

    status: str
    model_loaded: bool
    model_info: Dict[str, Any]
    cache_backend: str
    catalog_size: Dict[str, int]
    latency_stats: Dict[str, Any]
