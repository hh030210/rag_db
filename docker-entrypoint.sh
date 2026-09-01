#!/bin/bash
# docker-entrypoint.sh - 启动脚本
# 启动 MySQL + 应用（Milvus Lite 嵌入式，无需单独启动）

set -e

echo "========================================"
echo "  RAG Pipeline 容器启动中..."
echo "========================================"

# ===================== MySQL 启动 =====================
echo "[1/2] 启动 MySQL..."

# 初始化 MySQL 数据目录（首次启动）
if [ ! -d "/var/lib/mysql/mysql" ]; then
    echo "  首次启动，初始化 MySQL 数据目录..."
    mysql_install_db --user=mysql --datadir=/var/lib/mysql > /dev/null 2>&1
fi

# 启动 MySQL（后台）
mysqld --user=mysql \
    --datadir=/var/lib/mysql \
    --skip-networking=0 \
    --bind-address=0.0.0.0 \
    --port=3306 &

# 等待 MySQL 就绪
echo "  等待 MySQL 启动..."
for i in $(seq 1 30); do
    if mysqladmin ping -h 127.0.0.1 --silent 2>/dev/null; then
        echo "  MySQL 已就绪"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "[错误] MySQL 启动超时"
        exit 1
    fi
    sleep 1
done

# 配置 MySQL（设置密码、创建数据库）
echo "  配置 MySQL..."
mysql -u root <<EOF || true
ALTER USER 'root'@'localhost' IDENTIFIED BY '${MYSQL_ROOT_PASSWORD:-root}';
CREATE DATABASE IF NOT EXISTS ${MYSQL_DATABASE:-Main_index};
GRANT ALL PRIVILEGES ON *.* TO 'root'@'%' IDENTIFIED BY '${MYSQL_ROOT_PASSWORD:-root}' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON *.* TO 'root'@'localhost' IDENTIFIED BY '${MYSQL_ROOT_PASSWORD:-root}' WITH GRANT OPTION;
FLUSH PRIVILEGES;
EOF

echo "  MySQL 配置完成"

# ===================== 服务信息 =====================
echo ""
echo "========================================"
echo "  服务已启动"
echo "========================================"
echo "  MySQL:  127.0.0.1:3306"
echo "          用户: root"
echo "          密码: ${MYSQL_ROOT_PASSWORD:-root}"
echo "          数据库: ${MYSQL_DATABASE:-Main_index}"
echo ""
echo "  Milvus: /milvus_data (嵌入式，无需启动)"
echo "  模型:   /models (挂载或内置)"
echo "========================================"
echo ""

# ===================== 执行主命令 =====================
echo "启动应用: $@"
exec "$@"
