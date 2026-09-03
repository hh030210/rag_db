#!/usr/bin/env bash
set -euo pipefail

QUEUE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$QUEUE_DIR/.." && pwd)"
if [[ -x "/home/humq/envs/denoise_qa/bin/python" ]]; then
  PYTHON_BIN="/home/humq/envs/denoise_qa/bin/python"
else
  PYTHON_BIN="python3"
fi
exec "$PYTHON_BIN" "$QUEUE_DIR/run_nonhuman_queue.py" --code_root "$CODE_ROOT" "$@"
