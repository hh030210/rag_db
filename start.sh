#!/usr/bin/env bash
# ==============================================================================
# start.sh —— Linux 服务器启动入口
# 用法：
#     chmod +x start.sh
#     ./start.sh
# 环境变量（可选）：
#     HOST     监听地址，默认 0.0.0.0
#     PORT     监听端口，默认 8000
#     BGE_MODEL_PATH    BGE-M3 模型目录（默认 ./model/bge-m3）
#     PROMPT_OPT_FULL   是否打印完整结果（默认 0）
# ==============================================================================
set -e

export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-8000}

cd "$(dirname "$0")"

echo "============================================================"
echo "  Prompt Iteration Optimizer · 启动"
echo "  HOST=$HOST  PORT=$PORT"
echo "============================================================"

# 用 uvicorn 直接启动，--workers=1 复用 BGE 单例
exec uvicorn api_server:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers 1 \
    --log-level info