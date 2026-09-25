@echo off
REM =====================================================================
REM 一键跑通全流程：数据生成 -> 特征工程 -> 模型训练 -> 启动服务
REM 用法：双击本文件，或在 cmd 中执行 scripts\run_pipeline.bat
REM =====================================================================
setlocal
cd /d "%~dp0.."

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [错误] 未找到虚拟环境 %PY%
    echo 请先创建：python -m venv --system-site-packages .venv
    exit /b 1
)

REM Windows 控制台默认 GBK，中文日志需要显式指定 utf-8
set PYTHONIOENCODING=utf-8

echo.
echo === [1/4] 生成模拟数据 ===
"%PY%" -m recsys.data.generate_data || exit /b 1

echo.
echo === [2/4] 离线特征工程 ===
"%PY%" -m recsys.features.pipeline || exit /b 1

echo.
echo === [3/4] 训练 MMoE 多目标模型 ===
"%PY%" -m recsys.models.train || exit /b 1

echo.
echo === [4/4] 启动在线推荐服务 ===
echo 接口文档: http://127.0.0.1:8000/docs
"%PY%" -m recsys.services.main

endlocal
