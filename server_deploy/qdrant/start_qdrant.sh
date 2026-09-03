#!/bin/sh
set -eu

BASE_DIR=/home/humq/qdrant
CONFIG_FILE=/home/humq/server_deploy/qdrant/qdrant.yaml
DATA_DIR=/home/humq/qdrant_data
PID_FILE="$DATA_DIR/qdrant.pid"
LOG_FILE="$DATA_DIR/qdrant.log"

mkdir -p "$DATA_DIR/storage" "$DATA_DIR/snapshots"

if [ ! -x "$BASE_DIR/qdrant" ]; then
  echo "Qdrant binary not found or not executable: $BASE_DIR/qdrant" >&2
  exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Qdrant config not found: $CONFIG_FILE" >&2
  exit 1
fi

if [ -f "$PID_FILE" ]; then
  pid=$(cat "$PID_FILE" 2>/dev/null || true)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "Qdrant is already running, pid=$pid"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

nohup "$BASE_DIR/qdrant" --config-path "$CONFIG_FILE" \
  >>"$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "Qdrant started, pid=$(cat "$PID_FILE"), log=$LOG_FILE"
