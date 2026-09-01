#!/usr/bin/env bash
set -e

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# BGE-M3 模型路径软链已就绪
export BGE_MODEL_PATH="${BGE_MODEL_PATH:-/opt/search_service/models/bge-m3}"

# Qdrant 配置
export QDRANT_HOST="${QDRANT_HOST:-127.0.0.1}"
export QDRANT_PORT="${QDRANT_PORT:-6333}"

cd /opt/search_service/service
exec /mnt/userhome/liangyanjie/anaconda3/bin/python -m uvicorn \
    --app-dir /opt/search_service/service \
    search_api_server:app \
    --host 0.0.0.0 \
    --port "${PORT:-8100}" \
    --workers 1 \
    --log-level info
