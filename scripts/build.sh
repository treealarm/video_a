#!/usr/bin/env bash
set -euo pipefail

# Builds the analytics-worker docker image, ensuring the shared ta-deps base (owned by ta_vms —
# protobuf/grpc/spdlog/ffmpeg/openvino, prebuilt so this repo's Dockerfile doesn't recompile them)
# exists first.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TA_VMS_DIR="${TA_VMS_DIR:-$SCRIPT_DIR/../../ta_vms}"

[ -d "$TA_VMS_DIR" ] || { echo "ta_vms checkout not found: $TA_VMS_DIR (set TA_VMS_DIR)"; exit 1; }

# The encoder library, which the worker decodes through. The Dockerfile copies the working tree
# into the build context, so an uninitialised submodule reaches the image as an empty directory
# and fails inside the container at add_subdirectory(sve) -- a confusing place to learn that a
# clone was not recursive.
if [ ! -f "$SCRIPT_DIR/../sve/CMakeLists.txt" ]; then
    echo "=== Fetching the sve submodule ==="
    git -C "$SCRIPT_DIR/.." submodule update --init --recursive
fi

if ! docker image inspect ta-deps &>/dev/null; then
    echo "=== Building ta-deps ==="
    docker build -t ta-deps "$TA_VMS_DIR/ta-deps"
else
    echo "=== ta-deps already exists, skipping ==="
fi

echo "=== Building analytics-worker ==="
docker compose build
