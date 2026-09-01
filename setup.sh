#!/usr/bin/env bash
# ==============================================================================
# setup.sh - 在 Linux 服务器上一键准备依赖环境
# 用法：
#     chmod +x setup.sh
#     ./setup.sh
# 环境要求：
#     - Python 3.9+
#     - pip 已就绪（如果用 venv/conda，请先激活对应环境）
# ==============================================================================
set -e

echo "============================================================"
echo "  Prompt Iteration Optimizer - Linux 部署安装"
echo "============================================================"

PY=${PYTHON:-python3}
PIP=${PIP:-pip3}

echo "[1/5] 检查 Python 版本..."
$PY --version

echo "[2/5] 升级 pip..."
$PY -m pip install --upgrade pip

echo "[3/5] 安装 Web 框架 + DashScope（fastapi / uvicorn / pydantic / dashscope）..."
$PIP install -r requirements_api.txt

echo "[4/5] 安装 BGE 编码器依赖（按优先级二选一即可）..."
if ! $PY -m pip install FlagEmbedding; then
    echo "[警告] FlagEmbedding 安装失败，回退到 sentence-transformers"
    $PY -m pip install -r requirements_bge.txt
fi

echo "[5/5] 安装 numpy（向量运算）..."
$PY -m pip install -r requirements_numpy.txt

echo
echo "============================================================"
echo "  安装完成。运行 ./start.sh 启动服务。"
echo "============================================================"
