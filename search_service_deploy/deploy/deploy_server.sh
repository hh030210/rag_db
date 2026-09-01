#!/usr/bin/env bash
# deploy_server.sh
# 在阿里云 81.70.191.196 上执行：
#   bash deploy_server.sh
#
# 前置：
#   - 需要 docker + docker-compose-plugin 已装
#   - BGE-M3 模型需提前放 /root/mingqiang/model/bge-m3 （或 -m 指向其他路径）
#
# 该脚本做：
#   1. 创建目录
#   2. 拷贝本仓库 deploy/、service/ 到 /opt/search_service
#   3. 启动 Qdrant + 检索服务
#   4. 暴露 :8100 (search) / :6333 (qdrant)

set -euo pipefail

DEPLOY_DIR="/opt/search_service"
BGE_PATH="${BGE_PATH:-/root/mingqiang/model/bge-m3}"

echo "=== deploy_server.sh ==="
echo "目标目录: $DEPLOY_DIR"
echo "BGE-M3 路径: $BGE_PATH"
echo

# 0. docker 检查
if ! command -v docker >/dev/null; then
  echo "[ERROR] docker 未安装。请先安装 docker + docker-compose-plugin"
  exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
  echo "[ERROR] docker compose 未安装"
  exit 1
fi

# 1. 创建目录（首次）
mkdir -p "$DEPLOY_DIR"

# 2. 拷贝文件（假定当前在解包后的目录中运行此脚本）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cp -r "$ROOT_DIR/deploy" "$DEPLOY_DIR/"
cp -r "$ROOT_DIR/service" "$DEPLOY_DIR/"
mkdir -p "$DEPLOY_DIR/models"

# 3. BGE-M3 模型软链（如果存在且未在 models）
if [ -d "$BGE_PATH" ] && [ ! -e "$DEPLOY_DIR/models/bge-m3" ]; then
  ln -s "$BGE_PATH" "$DEPLOY_DIR/models/bge-m3"
  echo "  [OK] 已软链 BGE-M3: $BGE_PATH -> $DEPLOY_DIR/models/bge-m3"
elif [ -d "$BGE_PATH" ]; then
  echo "  [OK] BGE-M3 模型已存在"
else
  echo "[WARN] BGE-M3 路径 $BGE_PATH 不存在，请将模型放到该目录"
fi

# 4. 启服务
cd "$DEPLOY_DIR/deploy"
docker compose pull qdrant || true
docker compose up -d qdrant
echo "[1] 等待 Qdrant 启动..."
for i in {1..30}; do
  if curl -s http://127.0.0.1:6333/healthz >/dev/null 2>&1; then
    echo "  [OK] Qdrant ready"
    break
  fi
  sleep 2
done

echo "[2] 构建并启动 search 服务..."
docker compose build --no-cache search
docker compose up -d search

echo "[3] 当前状态:"
docker compose ps
echo
echo "[OK] 部署完成"
echo "  - Qdrant:     http://127.0.0.1:6333/healthz"
echo "  - Search API: http://127.0.0.1:8100/healthz"
echo "  - 调用示例:   curl -s -X POST http://127.0.0.1:8100/search -H 'Content-Type: application/json' -d '{\"query\":\"杭州西湖\",\"mode\":\"sem\",\"top_k\":8}'"
