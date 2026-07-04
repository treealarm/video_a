#!/usr/bin/env bash
set -euo pipefail

# Builds the analytics-worker docker image, ensuring the shared vms-deps base (owned by vms_rec —
# protobuf/grpc/spdlog/ffmpeg/openvino, prebuilt so this repo's Dockerfile doesn't recompile them)
# exists first.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VMS_REC_DIR="${VMS_REC_DIR:-$SCRIPT_DIR/../../vms_rec}"

[ -d "$VMS_REC_DIR" ] || { echo "vms_rec checkout not found: $VMS_REC_DIR (set VMS_REC_DIR)"; exit 1; }

if ! docker image inspect vms-deps &>/dev/null; then
    echo "=== Building vms-deps ==="
    docker build -t vms-deps "$VMS_REC_DIR/vms-deps"
else
    echo "=== vms-deps already exists, skipping ==="
fi

echo "=== Building analytics-worker ==="
docker compose build
