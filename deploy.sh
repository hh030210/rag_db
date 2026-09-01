#!/bin/bash
# deploy.sh - 一键部署脚本
# 用法: ./deploy.sh [OPTIONS]
#
# OPTIONS:
#   --model-dir PATH    服务器上模型目录 (默认: /data/models)
#   --port PORT         服务端口 (默认: 8000)
#   --name NAME         容器名 (默认: rag-pipeline)

set -e

# 默认值
MODEL_DIR="/data/models"
PORT=8000
CONTAINER_NAME="rag-pipeline"
IMAGE_NAME="rag-pipeline"
IMAGE_TAG="latest"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "  RAG Pipeline 部署配置"
echo "=========================================="
echo "  模型目录: $MODEL_DIR"
echo "  服务端口: $PORT"
echo "  容器名:   $CONTAINER_NAME"
echo "=========================================="

# 检查模型目录
if [ ! -d "$MODEL_DIR" ]; then
    echo "[错误] 模型目录不存在: $MODEL_DIR"
    exit 1
fi

# 检查模型是否存在
if [ ! -d "$MODEL_DIR/bge-m3" ]; then
    echo "[警告] 模型目录 $MODEL_DIR/bge-m3 不存在"
    echo "[提示] 请确保模型放在 $MODEL_DIR/bge-m3/"
fi

# 停止并删除旧容器
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[1/4] 停止并删除旧容器..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
fi

# 构建镜像
echo "[2/4] 构建 Docker 镜像..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

# 启动容器
echo "[3/4] 启动容器..."
docker run -d \
    --name $CONTAINER_NAME \
    --restart unless-stopped \
    -p ${PORT}:8000 \
    -v ${MODEL_DIR}:/models \
    -v $(pwd)/db_config.yaml:/app/db_config.yaml \
    -e MODEL_PATH=/models \
    ${IMAGE_NAME}:${IMAGE_TAG} \
    python pipeline.py

# 等待容器启动
echo "[4/4] 等待服务启动..."
sleep 5

# 检查容器状态
if docker ps | grep -q $CONTAINER_NAME; then
    echo ""
    echo "=========================================="
    echo "  部署成功!"
    echo "=========================================="
    echo "  容器名:   $CONTAINER_NAME"
    echo "  服务端口: $PORT"
    echo "  模型目录: $MODEL_DIR"
    echo ""
    echo "查看日志: docker logs -f $CONTAINER_NAME"
else
    echo "[错误] 容器启动失败，查看日志:"
    docker logs $CONTAINER_NAME
    exit 1
fi
