#!/usr/bin/env bash
# =====================================================================
# 一键跑通全流程：数据生成 -> 特征工程 -> 模型训练 -> 启动服务
# 用法：bash scripts/run_pipeline.sh
# =====================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

PY=".venv/Scripts/python.exe"
[ -x "$PY" ] || PY=".venv/bin/python"          # 兼容 Linux/macOS
if [ ! -x "$PY" ]; then
    echo "[错误] 未找到虚拟环境，请先创建：python -m venv --system-site-packages .venv"
    exit 1
fi
export PYTHONIOENCODING=utf-8

echo
echo "=== [1/4] 生成模拟数据 ==="
"$PY" -m recsys.data.generate_data

echo
echo "=== [2/4] 离线特征工程 ==="
"$PY" -m recsys.features.pipeline

echo
echo "=== [3/4] 训练 MMoE 多目标模型 ==="
"$PY" -m recsys.models.train

echo
echo "=== [4/4] 启动在线推荐服务 ==="
echo "接口文档: http://127.0.0.1:8000/docs"
"$PY" -m recsys.services.main
