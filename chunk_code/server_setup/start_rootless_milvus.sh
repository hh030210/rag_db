#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/home/humq/milvus_denoise}"
RUNTIME_DIR="$ROOT_DIR/run"
DOCKER_SOCKET="$RUNTIME_DIR/docker.sock"
COMPOSE_FILE="${COMPOSE_FILE:-$ROOT_DIR/server_setup/docker-compose.milvus.yml}"

mkdir -p "$ROOT_DIR/docker-data" "$ROOT_DIR/docker-exec" "$RUNTIME_DIR" "$ROOT_DIR/logs"
chmod 700 "$RUNTIME_DIR"

if ! command -v newuidmap >/dev/null 2>&1 || ! command -v newgidmap >/dev/null 2>&1; then
  echo "Rootless Docker requires newuidmap and newgidmap (install the uidmap package as an administrator)." >&2
  exit 2
fi

export XDG_RUNTIME_DIR="$RUNTIME_DIR"
export DOCKER_HOST="unix://$DOCKER_SOCKET"
export DOCKERD_ROOTLESS_ROOTLESSKIT_NET="slirp4netns"
export DOCKERD_ROOTLESS_ROOTLESSKIT_PORT_DRIVER="slirp4netns"

if ! docker info >/dev/null 2>&1; then
  if [ -f "$RUNTIME_DIR/dockerd.pid" ] && kill -0 "$(cat "$RUNTIME_DIR/dockerd.pid")" 2>/dev/null; then
    echo "dockerd-rootless is starting; waiting for $DOCKER_SOCKET"
  else
    nohup dockerd-rootless.sh \
      --host="$DOCKER_HOST" \
      --data-root="$ROOT_DIR/docker-data" \
      --exec-root="$ROOT_DIR/docker-exec" \
      --pidfile="$RUNTIME_DIR/dockerd.pid" \
      >"$ROOT_DIR/logs/dockerd.log" 2>&1 &
    echo $! > "$RUNTIME_DIR/dockerd.launcher.pid"
  fi
fi

for _ in $(seq 1 60); do
  if docker info >/dev/null 2>&1; then
    echo "Rootless Docker is ready: $DOCKER_HOST"
    exec docker compose -f "$COMPOSE_FILE" up -d
  fi
  sleep 2
done

echo "Rootless Docker failed to start. Check: $ROOT_DIR/logs/dockerd.log" >&2
exit 1
