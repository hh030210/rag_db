#!/usr/bin/env bash
# ==============================================================================
# smoke_test.sh —— 启动后冒烟测试
# ==============================================================================
set -e

HOST=${HOST:-81.70.191.196}
PORT=${PORT:-8000}
BASE="http://${HOST}:${PORT}"

echo "============================================================"
echo "  冒烟测试  ${BASE}"
echo "============================================================"

echo "[1/2] 健康检查..."
code=$(curl -s -o /dev/null -w "%{http_code}" "${BASE}/healthz")
if [ "$code" != "200" ]; then
    echo "  FAIL: healthz 返回 $code"
    exit 1
fi
echo "  OK"

echo "[2/2] /optimize 场景级..."
resp=$(curl -s -X POST "${BASE}/optimize" \
    -H "Content-Type: application/json" \
    -d '{"api_key":"test","query":"北京三日游推荐"}')
if echo "$resp" | grep -q '"ok":true'; then
    echo "  OK"
else
    echo "  FAIL: $resp"
    exit 1
fi

echo
echo "============================================================"
echo "  全部通过"
echo "============================================================"
