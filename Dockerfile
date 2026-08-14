# RAG Pipeline Dockerfile (精简版 - 调用外部 Milvus/MySQL)
# ============================
# 镜像大小约 500MB（只包含 Python 依赖和代码）
# 需要外部 Milvus 和 MySQL 服务

FROM python:3.10-slim

WORKDIR /app

# ===================== 安装系统依赖 =====================
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ===================== 安装 Python 依赖 =====================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# ===================== 环境变量 =====================
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# ===================== 启动命令 =====================
CMD ["python", "api_server.py"]
