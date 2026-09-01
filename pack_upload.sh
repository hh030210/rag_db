#!/usr/bin/env bash
# ==============================================================================
# pack_upload.sh —— 把要上传到 Linux 服务器的文件打成 zip
# 用法：
#     chmod +x pack_upload.sh
#     ./pack_upload.sh
# 产出：
#     ./dist/prompt_iteration_optimizer_<时间戳>.zip
# 注意：
#     - 训练产物路径 "code1/chapter3_backup/..." 必须保持相对路径，
#       上传后保持同样的目录布局。
# ==============================================================================
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

OUT_DIR="$ROOT/dist"
mkdir -p "$OUT_DIR"
TS=$(date +%Y%m%d_%H%M%S)
ZIP_FILE="$OUT_DIR/prompt_iteration_optimizer_${TS}.zip"

# 待打包的核心文件
CORE_FILES=(
    "prompt_iteration_optimizer.py"
    "prompt_iteration_service.py"
    "api_server.py"
    "requirements_api.txt"
    "requirements_bge.txt"
    "requirements_numpy.txt"
    "setup.sh"
    "start.sh"
    "smoke_test.sh"
    "README_api.md"
    "DEPLOY.md"
)

# 训练产物（如果存在）
TRAIN_DIR="$ROOT/code1/chapter3_backup/codes/bylw_rag/new_experiments"

echo "============================================================"
echo "  打包核心文件 + 训练产物 → $ZIP_FILE"
echo "============================================================"

# 先把核心文件放进临时结构
TMP=$(mktemp -d)
mkdir -p "$TMP/core"
for f in "${CORE_FILES[@]}"; do
    if [ -f "$ROOT/$f" ]; then
        cp "$ROOT/$f" "$TMP/core/"
    else
        echo "[警告] $f 不存在，跳过"
    fi
done

# 训练产物存在则一并打包
if [ -d "$TRAIN_DIR" ]; then
    cp -r "$TRAIN_DIR" "$TMP/core/code1_train_data"
    echo "[信息] 已包含训练产物: $TRAIN_DIR"
else
    echo "[警告] 训练产物目录 $TRAIN_DIR 不存在，"
    echo "       上传后请把对应文件放到 ./code1/chapter3_backup/... 路径下。"
fi

# 打包
(cd "$TMP" && zip -r "$ZIP_FILE" core -q)
echo "[完成] $ZIP_FILE"

# 列文件清单
echo
echo "打包内容："
unzip -l "$ZIP_FILE" | tail -n +4

rm -rf "$TMP"