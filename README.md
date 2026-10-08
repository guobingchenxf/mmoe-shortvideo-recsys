# 基于 MMoE 的实时短视频推荐引擎

一个**端到端可运行**的短视频推荐系统 Demo：从模拟日志生成、离线特征工程、MMoE 多目标建模，
到 FastAPI 在线推理服务与 Docker 部署。核心是**多目标优化**（同时预估 点击 / 点赞 / 完播率）
与**工程落地**（召回-精排-重排链路、防特征泄漏、低延时服务）。

> 大部分代码为手写实现（MMoE、多任务损失、AUC/GAUC 指标均为自行实现，非调包），LLM实现错误检查与迭代
> 配有完整的三方 ablation 实验与单元测试。

---

## 目录

- [项目亮点](#项目亮点)
- [系统架构](#系统架构)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [接口文档](#接口文档)
- [模型效果](#模型效果)
- [性能压测](#性能压测)
- [关键设计](#关键设计)
- [Docker 部署](#docker-部署)
- [测试](#测试)
- [局限与后续改进](#局限与后续改进)

---

## 项目亮点

| 维度 | 说明 |
|---|---|
| **多目标建模** | 手写 MMoE（Shared Bottom + N Expert + N Gate + N Tower），非调包 |
| **动态损失权重** | 实现 Uncertainty Weighting（Kendall 2018），任务权重自动学习 |
| **完整 ablation** | 三方对比：单任务独立建模 ×3 / Shared-Bottom / MMoE |
| **防特征泄漏** | 时间切分 + 训练集拟合归一化 + 历史特征只用过去（有测试守住） |
| **大规模稀疏特征** | 稳定哈希编码 + 稀疏 Embedding + SparseAdam + 独立学习率 |
| **可解释推荐** | 每条结果带召回来源与推荐理由，以及三个目标的独立预测值 |
| **零依赖可跑** | 无 GPU、无 Redis、无 Docker 也能一键跑通（自动降级内存缓存） |
| **工程细节** | 线程超订调优、模型预热、分段延时统计、冷启动与边界防御 |

---

## 系统架构

### 离线链路

```
 生成模拟日志            特征工程                模型训练              离线评估
┌──────────────┐   ┌──────────────────┐   ┌────────────────┐   ┌────────────────┐
│ generate_data│   │ FeatureColumn    │   │ MMoE           │   │ evaluate       │
│  · 隐因子模型 │──▶│  · dense 归一化   │──▶│  · SharedBottom│──▶│  · 单任务 ×3    │
│  · 截距校准   │   │  · 稳定哈希编码   │   │  · 8 × Expert  │   │  · SharedBottom │
│  · 标签相关性 │   │  · 序列 padding   │   │  · 3 × Gate    │   │  · MMoE         │
└──────────────┘   │  · 时间切分/防泄漏 │   │  · 3 × Tower   │   └────────────────┘
                   └──────────────────┘   │  · 不确定性加权 │
   data/raw/*.csv    data/processed/*.parquet  artifacts/models/best_mmoe.pt
```

### 在线链路（漏斗式）

```
 POST /recommend {user_id, context_features, top_k}
        │
        ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 1. 取用户行为序列   CacheClient (Redis / 内存)            │
 ├──────────────────────────────────────────────────────────┤
 │ 2. 多路召回        同类目兴趣 · 关注作者 · 热门  → 300 条   │
 ├──────────────────────────────────────────────────────────┤
 │ 3. 特征组装        dense + sparse + multi-hot + sequence │
 ├──────────────────────────────────────────────────────────┤
 │ 4. MMoE 精排       批量前向 → pctr / plike / pwatch       │
 ├──────────────────────────────────────────────────────────┤
 │ 5. 融合排序        0.6·pctr + 0.3·plike + 0.1·pwatch      │
 ├──────────────────────────────────────────────────────────┤
 │ 6. 重排与解释      生成推荐理由，返回 Top-K               │
 └──────────────────────────────────────────────────────────┘
        │
        ▼
 {video_list: [{video_id, score, reason, recall_source, task_scores}], latency_ms}
```

---

## 目录结构

```
mmoe-shortvideo-recsys/
├── README.md                       项目总览
├── requirements.txt                依赖清单
├── configs/config.yaml             全局配置（数据规模 / 模型超参 / 服务参数）
│
├── recsys/                         ── 核心代码 ──
│   ├── config.py                   yaml -> 强类型 dataclass
│   ├── data/
│   │   ├── schema.py               三张表的字段常量与多任务定义
│   │   └── generate_data.py        模拟数据生成（隐因子 + 截距校准）
│   ├── features/
│   │   ├── feature_column.py       特征声明层 FeatureColumn / FeatureSchema
│   │   ├── preprocess.py           哈希编码 / 归一化 / multi-hot / 序列构造
│   │   └── pipeline.py             离线特征流水线（时间切分 + 防泄漏）
│   ├── models/
│   │   ├── dataset.py              parquet -> 张量，Dataset/DataLoader
│   │   ├── mmoe.py                 EmbeddingInputLayer / Gate / MMoE / 单任务基线
│   │   ├── loss.py                 MultiTaskLoss（固定权重 / Uncertainty Weighting）
│   │   ├── metrics.py              AUC / GAUC / RMSE（纯 numpy）
│   │   ├── train.py                训练循环 + EarlyStopping + 模型打包
│   │   └── evaluate.py             三方对比实验
│   ├── services/
│   │   ├── schemas.py              Pydantic 接口契约
│   │   ├── redis_client.py         CacheClient 抽象（Redis / 内存）
│   │   ├── model_server.py         模型加载、特征编码、批量推理、预热
│   │   ├── ranking.py              召回 + 特征组装 + 融合排序 + 推荐理由
│   │   └── main.py                 FastAPI 应用
│   └── utils/                      日志 / 计时 / IO
│
├── tests/                          28 个单元测试
├── scripts/run_pipeline.bat|.sh    一键跑通脚本
├── docker/Dockerfile|compose      容器化部署
├── docs/项目详解.md                原理、实验分析、踩坑记录
├── data/{raw,processed}/           数据产物（不入库）
└── artifacts/{models,logs}/        模型与训练日志（不入库）
```

---

## 快速开始

### 环境要求

- Python 3.10
- 无需 GPU（CPU 即可训练与推理）
- 无需 Redis（未配置时自动使用内存缓存）

### 步骤

```bash
cd D:\mmoe-shortvideo-recsys

# 0. 创建conda虚拟环境
python -m venv --system-site-packages .venv
.venv\Scripts\python -m pip install -r requirements.txt

# 1. 生成模拟数据（30 万条曝光，约 10 秒）
.venv\Scripts\python -m recsys.data.generate_data

# 2. 离线特征工程（约 30 秒）
.venv\Scripts\python -m recsys.features.pipeline

# 3. 训练 MMoE（CPU 约 6 分钟）
.venv\Scripts\python -m recsys.models.train

# 4. 启动推荐服务
.venv\Scripts\python -m recsys.services.main
#   → 接口文档 http://127.0.0.1:8000/docs

# 5. 离线基线对比（可选，约 15 分钟）
.venv\Scripts\python -m recsys.models.evaluate
```
---

## 接口文档

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_info": {
    "model": "MMoE(experts=8, tasks=3, input_dim=93, sparse_emb=True, params=3,634,515)",
    "labels": ["click", "like", "watch_time_ratio"],
    "offline_metrics": {"click_auc": 0.6247, "click_gauc": 0.5853, "like_auc": 0.6657,
                        "like_gauc": 0.5756, "watch_time_ratio_rmse": 0.2968}
  },
  "cache_backend": "in-memory",
  "catalog_size": {"users": 5000, "videos": 20000},
  "latency_stats": {"count": 68, "p50": 51.8, "p90": 66.5, "p95": 73.6, "p99": 74.9}
}
```

### `POST /recommend`

请求：

```json
{
  "user_id": 123,
  "top_k": 5,
  "context_features": {"hour": 21, "device": 0, "source": 2}
}
```

响应（真实返回）：

```json
{
  "user_id": 123,
  "is_cold_start": false,
  "candidate_count": 300,
  "latency_ms": 50.2,
  "latency_breakdown": {"context": 0.1, "recall": 0.4, "features": 14.2,
                        "inference": 30.5, "rank": 5.0},
  "video_list": [
    {
      "video_id": 5356,
      "score": 0.1325,
      "reason": "你最近常看「4」类内容",
      "recall_source": "同类目兴趣",
      "task_scores": {"click": 0.1315, "like": 0.03527, "watch_time_ratio": 0.430}
    }
  ]
}
```

字段说明：

- `recall_source`：该候选来自哪一路召回（同类目兴趣 / 关注作者 / 热门推荐）
- `task_scores`：三个目标的独立预测值，便于排查"为什么这条排前面"
- `latency_breakdown`：分段耗时，定位延时瓶颈用
- `is_cold_start`：未知用户走热门兜底召回

---

## 模型效果

### 数据规模

| 项目 | 值 |
|---|---|
| 曝光样本 | 300,000（训练 240,000 / 验证 60,000，按时间切分） |
| 用户 / 视频 / 创作者 | 5,000 / 20,000 / 2,000 |
| 点击率 / 点赞率 | 0.0501 / 0.0100（生成时做了截距校准，精确命中） |
| 完播率 | P10/P50/P90 = 0.027 / 0.336 / 0.839（长尾） |

### 特征（23 个）

| 类型 | 数量 | 内容 |
|---|---|---|
| dense | 13 | 年龄、时长、热度、新鲜度、注册天数、活跃度、城市等级、小时/星期的 sin-cos 编码、用户历史点击率/曝光数 |
| sparse | 8 | user_id、video_id(哈希)、author_id、category、gender、device、source、user×category 交叉(哈希) |
| multi_hot | 1 | genres（视频标签，哈希后 masked-mean 池化） |
| sequence | 1 | user_seq（最近点击的 50 个视频，倒序、PAD 补齐） |

### 训练配置

`embedding_dim=8`，`shared_bottom=[256]`，`8 experts=[128,64]`，`towers=[64,32]`，
batch 2048，Adam(lr=1e-3) + SparseAdam(embedding_lr=5e-3)，dropout 0.25，EarlyStopping(patience=3)

### 验证集指标

| 任务 | AUC | GAUC | RMSE |
|---|---|---|---|
| click | 0.6247 | 0.5853 | — |
| like | 0.6657 | 0.5756 | — |
| 完播率 | — | — | 0.2968 |

### 三方对比（同一数据 / 同一特征 / 同一训练配置）

| 任务 | 指标 | 单任务独立 ×3 | Shared-Bottom | MMoE(8 专家) |
|---|---|---|---|---|
| click | AUC | 0.6225 | 0.6219 | **0.6247** |
| click | GAUC | 0.5876 | **0.5912** | 0.5853 |
| like | AUC | **0.6675** | 0.6558 | 0.6657 |
| like | GAUC | 0.5590 | **0.5855** | 0.5756 |
| 完播率 | RMSE | 0.2975 | 0.2977 | **0.2968** |

**结论（如实说明）：差异都在 ±0.003 以内，落在噪声范围内（3000 个正样本对应 AUC 标准误约 0.005~0.008），
不能宣称 MMoE 显著胜出。** 原因与 MMoE 原论文的结论一致 —— 任务相关性越高，MoE 相对硬共享的增益越小；
本项目三个目标都由同一隐因子驱动，且交互稀疏（0.024% 密度）导致各模型撞到同一效果天花板。

两个可靠结论：
1. **MMoE 用 1 个模型达到了 3 个独立模型的效果**（平均 AUC 0.6452 vs 0.6450），参数与工程量更省；
2. **Shared-Bottom 在 like 上出现负迁移**（0.6558 vs 单任务 0.6675），MMoE 恢复到 0.6657。

详细的实验分析与改进方向见 [docs/项目详解.md](docs/项目详解.md)。

---

## 性能压测

召回 300 条候选，连续 60 次请求（单机 CPU，i5-11300H）：

| 指标 | min | P50 | P90 | P95 | P99 |
|---|---|---|---|---|---|
| **模型推理** | 15ms | **32ms** | 40ms | **45ms** | 52ms |
| 端到端（服务端） | 26ms | 52ms | 65ms | 74ms | 75ms |

目标「单次推理 < 50ms」在 P95 上达成。

### 线程数对延时的影响（重要）

| `TORCH_NUM_THREADS` | P50 | max |
|---|---|---|
| **1** | 44.6ms | **50.7ms** |
| 2 | 43.3ms | 317ms |
| 4 | 33.7ms | 308ms |
| 8 | **1416ms** | 2112ms |

本机 8 逻辑核上开 4/8 线程，小批量推理会出现 OpenMP 线程超订，延迟抖动放大 30 倍。
推荐接口是"小批量 + 低延时"负载，**限制为 1 线程反而更稳更快**，故默认 `TORCH_NUM_THREADS=1`。

---

## 关键设计

### 1. 特征声明层：从机制上杜绝线上线下不一致

`FeatureSchema` 是一份可序列化的特征说明书。离线训练与在线推理**读同一份 spec**
（哈希桶数、序列长度、归一化统计量都来自训练时保存的 `bundle`），
从根本上避免 Training-Serving Skew。

### 2. 三重防泄漏

| 泄漏类型 | 做法 |
|---|---|
| 标签泄漏 | 历史点击率、行为序列严格只用当前样本之前的数据（先记录、后更新），有单测覆盖 |
| 统计量泄漏 | 归一化 mean/std 只在训练集 fit，再 transform 验证集 |
| 时间泄漏 | 按时间切分 train/valid，模拟"用过去预测未来" |

### 3. 稳定哈希，而不是 Python 内置 `hash()`

Python 对 str 的 `hash()` 带随机盐（`PYTHONHASHSEED`），同一字符串在不同进程结果不同 ——
离线训练与线上服务若用不同进程，特征索引会静默错位。本项目统一用 `blake2b` 做稳定哈希。
（当前 video_id 哈希碰撞率 9.19%，日志中会输出该诊断。）

### 4. 稀疏 Embedding + SparseAdam + 独立学习率

4 张 10 万级词表的 Embedding 共 325 万参数。若用普通 Adam，每步都要更新整张表
（实测占单步耗时 45%）。改为 `nn.Embedding(sparse=True)` + `SparseAdam` 只更新命中的行，
提速约 2 倍；同时给 Embedding 单独的更大学习率（稀疏行更新频次远低于稠密层）。

### 5. 冷启动与边界防御

未知用户映射到合法 Embedding 槽位并使用全站中位数画像；越界的上下文特征
（如 `device=999`）在接口边界统一取模裁剪 —— 只在系统边界做校验，不在内部重复校验。

---

## Docker 部署

```bash
cd docker
docker compose up --build
```

会启动两个容器：

- `recsys-api`：推荐服务（8000 端口）
- `recsys-redis`：行为序列缓存（配了 `REDIS_URL`，服务自动切换为 Redis 后端）

---

## 测试

```bash
.venv\Scripts\python -m pytest tests/ -q
# 28 passed
```

| 文件 | 覆盖内容 |
|---|---|
| `test_data.py` | 截距校准精度、标签分布命中目标、点赞以点击为前提 |
| `test_features.py` | 哈希稳定性与取值范围、归一化一致性、**行为序列/历史统计的防泄漏** |
| `test_model.py` | MMoE 输出形状、门控 softmax 归一化、稀疏/稠密参数划分、稀疏梯度、损失与指标 |
| `test_service.py` | 接口契约、排序单调性、冷启动、越界上下文、参数校验（422） |

---

## 局限与后续改进

1. **效果天花板受限于数据稀疏度**：30 万曝光铺在 5000×20000 的 ID 空间上密度仅 0.024%，
   模型学不到隐因子便转向记忆训练集（表现为 epoch 1 即最优）。改进：提升交互密度、
   negative sampling 采样、对 ID 做 dropout 正则。
2. **MoE 优势未被显著验证**：需要降低任务相关性（注入冲突目标）并做多种子重复实验。
3. **召回层较简单**：目前是规则化多路召回。生产环境应上双塔 + 向量检索（Faiss/Milvus）。
4. **未实现粗排层**：真实链路是"召回 → 粗排 → 精排 → 重排"，本项目省略了粗排。
5. **单机训练**：未引入参数服务器 / 分布式训练。
6. **重排策略单一**：未实现多样性（DPP）、打散、频率控制等。


---

## 参考资料

- Ma et al. *Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts*. KDD 2018.
- Kendall et al. *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics*. CVPR 2018.
- 王喆. 《深度学习推荐系统》.
- 项亮. 《推荐系统实践》.
