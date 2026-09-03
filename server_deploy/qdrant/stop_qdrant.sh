#!/bin/sh
set -eu

PID_FILE=/home/humq/qdrant_data/qdrant.pid

if [ ! -f "$PID_FILE" ]; then
  echo "Qdrant pid file not found"
  exit 0
fi

pid=$(cat "$PID_FILE")
if kill -0 "$pid" 2>/dev/null; then
  kill -TERM "$pid"
  echo "Sent SIGTERM to Qdrant pid=$pid"
else
  echo "Qdrant pid=$pid is not running"
fi
rm -f "$PID_FILE"
