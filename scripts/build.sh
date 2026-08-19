#!/usr/bin/env bash
set -euo pipefail

# Builds the analytics-worker docker image, ensuring the shared ta-deps base (owned by ta_vms —
# protobuf/grpc/spdlog/ffmpeg/openvino, prebuilt so this repo's Dockerfile doesn't recompile them)
# exists first.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TA_VMS_DIR="${TA_VMS_DIR:-$SCRIPT_DIR/../../ta_vms}"

[ -d "$TA_VMS_DIR" ] || { echo "ta_vms checkout not found: $TA_VMS_DIR (set TA_VMS_DIR)"; exit 1; }

if ! docker image inspect ta-deps &>/dev/null; then
    echo "=== Building ta-deps ==="
    docker build -t ta-deps "$TA_VMS_DIR/ta-deps"
else
    echo "=== ta-deps already exists, skipping ==="
fi

echo "=== Building analytics-worker ==="
docker compose build
