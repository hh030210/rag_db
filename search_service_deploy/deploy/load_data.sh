#!/usr/bin/env bash
# load_data.sh
# 在 81.70.191.196 上灌库：
#
#   # 1. 把 chunks jsonl 拷贝到 /opt/search_service/data/chunks.jsonl
#   scp chunks.jsonl root@81.70.191.196:/opt/search_service/data/
#   
#   # 2. 启一次 loader
#   cd /opt/search_service/deploy
#   docker compose --profile init run --rm loader \
#       --input /data/chunks.jsonl \
#       --chunk-collection unified_corpus \
#       --dim-tags-collection dimension_tags
#
# 假定：
#   - jsonl 中每行为: {"chunk_id":"...","chunk_text":"...","doc_title":"...","chunk_gen_title":"..."}
#   - 若还有 dim_tags.jsonl，则可继续 --dim-tags-input 灌 dimension_tags collection

set -euo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-/opt/search_service/deploy}"
CHUNK_COLLECTION="${CHUNK_COLLECTION:-unified_corpus}"
DIM_TAGS_COLLECTION="${DIM_TAGS_COLLECTION:-dimension_tags}"
INPUT="${1:-}"
VECTOR_DIM="${VECTOR_DIM:-1024}"

if [ -z "$INPUT" ]; then
  echo "用法: bash load_data.sh /data/chunks.jsonl [--dim-tags-input /data/dim_tags.jsonl]"
  exit 1
fi

cd "$DEPLOY_DIR"

if [ ! -f "$INPUT" ]; then
  echo "[ERROR] 输入文件不存在: $INPUT"
  exit 1
fi

ARGS=(
  --input "$INPUT"
  --chunk-collection "$CHUNK_COLLECTION"
  --dim-tags-collection "$DIM_TAGS_COLLECTION"
  --vector-dim "$VECTOR_DIM"
  --bge-model /models/bge-m3
)

echo "[load_data] 启动 loader..."
docker compose --profile init run --rm loader "${ARGS[@]}"

echo "[OK] 灌库完成"
docker compose exec qdrant wget -q -O - http://localhost:6333/collections | head -200 || true
