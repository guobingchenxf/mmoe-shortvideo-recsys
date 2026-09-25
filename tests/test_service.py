"""在线服务接口测试。

依赖已训练好的模型与特征数据，因此整组测试会在缺少产物时整体跳过：
    python -m recsys.models.train        # 先生成模型
    pytest tests/test_service.py -v

注意：TestClient 会触发 FastAPI 的 lifespan，即真实加载模型 + 物料库 + 用户画像
（本机约 30~60 秒），因此这组测试相对较慢。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from recsys.config import PROJECT_ROOT

BUNDLE = PROJECT_ROOT / "artifacts" / "models" / "best_mmoe.pt"
TRAIN_PARQUET = PROJECT_ROOT / "data" / "processed" / "train.parquet"
USERS_CSV = PROJECT_ROOT / "data" / "raw" / "users.csv"

pytestmark = pytest.mark.skipif(
    not (BUNDLE.exists() and TRAIN_PARQUET.exists() and USERS_CSV.exists()),
    reason="缺少已训练模型或数据，请先执行数据生成 / 特征工程 / 模型训练",
)


@pytest.fixture(scope="module")
def client():
    """整个模块共用一个应用实例，避免重复加载模型。"""
    from recsys.services.main import app

    with TestClient(app) as c:
        yield c


def test_health_reports_model_and_cache(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200

    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["catalog_size"]["videos"] > 0
    assert data["cache_backend"] in {"in-memory", "redis"}
    # 模型信息里应带上离线指标，便于线上确认加载的是哪一版
    assert "offline_metrics" in data["model_info"]


def test_recommend_contract(client: TestClient) -> None:
    """校验响应契约与推荐结果的完备性。"""
    resp = client.post("/recommend", json={"user_id": 123, "top_k": 5})
    assert resp.status_code == 200

    data = resp.json()
    assert data["user_id"] == 123
    assert len(data["video_list"]) == 5
    assert data["candidate_count"] > 0
    assert data["latency_ms"] > 0

    scores = [item["score"] for item in data["video_list"]]
    assert scores == sorted(scores, reverse=True), "推荐列表必须按分数降序"

    for item in data["video_list"]:
        assert item["video_id"] >= 0
        assert item["reason"], "每条推荐都应带推荐理由"
        assert set(item["task_scores"]) == {"click", "like", "watch_time_ratio"}
        assert 0.0 <= item["task_scores"]["click"] <= 1.0
        assert 0.0 <= item["task_scores"]["like"] <= 1.0


def test_recommend_cold_start_user(client: TestClient) -> None:
    """未知用户必须走冷启动分支且不能报错（历史上这里曾因 id 越界抛 500）。"""
    resp = client.post("/recommend", json={"user_id": 10**9, "top_k": 3})

    assert resp.status_code == 200
    data = resp.json()
    assert data["is_cold_start"] is True
    assert len(data["video_list"]) == 3


def test_recommend_out_of_range_context_is_clamped(client: TestClient) -> None:
    """越界的上下文特征应被兜底裁剪，而不是让 Embedding 查表越界。"""
    resp = client.post(
        "/recommend",
        json={"user_id": 77, "top_k": 2, "context_features": {"hour": 99, "device": 999, "source": -5}},
    )
    assert resp.status_code == 200
    assert len(resp.json()["video_list"]) == 2


def test_recommend_rejects_invalid_top_k(client: TestClient) -> None:
    """参数校验交给 Pydantic：top_k 越界应返回 422 而不是 500。"""
    assert client.post("/recommend", json={"user_id": 1, "top_k": 0}).status_code == 422
    assert client.post("/recommend", json={"user_id": 1, "top_k": 999}).status_code == 422
