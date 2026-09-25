# =====================================================================
# 基于 MMoE 的实时短视频推荐引擎 —— 在线服务镜像
#
# 构建（在项目根目录执行）：
#   docker build -f docker/Dockerfile -t mmoe-recsys:latest .
# 或直接用 compose：
#   cd docker && docker compose up --build
# =====================================================================
FROM python:3.10-slim

# PYTHONDONTWRITEBYTECODE: 不写 .pyc，保持镜像干净
# PYTHONUNBUFFERED:       日志实时输出（否则容器日志会攒着不刷新）
# PYTHONIOENCODING:       中文日志必须显式 utf-8，否则容器内默认编码可能报错
# TORCH_NUM_THREADS:      小批量推理下"少线程更稳更快"，详见 README 压测章节
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    TORCH_NUM_THREADS=1

WORKDIR /app

# 系统依赖：
#   libgomp1 —— PyTorch / numpy 依赖的 OpenMP 运行时（slim 镜像默认没有）
#   curl     —— 供 HEALTHCHECK 探活使用
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖、再拷代码 —— 代码改动不会让依赖层缓存失效，显著加快重复构建
COPY requirements.txt .
# 镜像里没有宿主机的 torch，需要显式安装 CPU 版；放在 requirements 之前避免被依赖解析覆盖
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY recsys/ ./recsys/
COPY configs/ ./configs/
COPY tests/ ./tests/
COPY requirements.txt README.md ./

EXPOSE 8000

# 模型加载 + 物料库构建需要几十秒，start-period 给足，避免刚起就被判定为不健康
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["python", "-m", "recsys.services.main"]
