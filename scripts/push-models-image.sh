#!/usr/bin/env bash
set -euo pipefail

# Publishes models/ as treealarm/analytics-models:<tag> and records <tag> in models.tag, for CI to
# bake into the analytics-worker image. Run it whenever models/ changes, then commit models.tag:
# the models are not in git (licensing, and size), so a CI checkout has none of its own and fails
# with a pointer here when the tag it is given is not published.
#
# The tag is a hash of the contents of models/, so an image already published under it is the right
# one by construction -- re-running this with nothing changed pushes nothing.
#
# Authentication: run `docker login` beforehand, or export DOCKERHUB_USERNAME + DOCKERHUB_TOKEN
# in the environment — credentials are never stored in this repository.

export DOCKER_BUILDKIT=1

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$REPO_DIR/models"
NAMESPACE="treealarm"
REGISTRY="docker.io"

# A symlink reaches the build context as a link, not as the file it points at, and lands in the
# image dangling -- the worker then reports that model as absent. A subdirectory is not a model the
# worker loads. Both are mistakes to fix in models/, not to publish.
odd="$(find "$MODELS_DIR" -mindepth 1 ! -type f)"
if [ -n "$odd" ]; then
    echo "models/ must hold regular files only; move these out first:" >&2
    echo "$odd" >&2
    exit 1
fi
if ! find "$MODELS_DIR" -type f ! -name .gitkeep | grep -q .; then
    echo "models/ is empty -- there is nothing to publish" >&2
    exit 1
fi

TAG="$(cd "$MODELS_DIR" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -c1-12)"
IMAGE="$NAMESPACE/analytics-models:$TAG"

if [ -n "${DOCKERHUB_TOKEN:-}" ]; then
    echo "${DOCKERHUB_TOKEN}" | docker login "$REGISTRY" -u "${DOCKERHUB_USERNAME:?DOCKERHUB_USERNAME must be set when DOCKERHUB_TOKEN is used}" --password-stdin
fi

if docker buildx imagetools inspect "$IMAGE" &>/dev/null; then
    echo "=== $IMAGE is already published ==="
else
    echo "=== Building $IMAGE ==="
    docker build --target models -t "$IMAGE" "$REPO_DIR"
    echo "=== Pushing $IMAGE ==="
    docker push "$IMAGE"
fi

echo "$TAG" > "$REPO_DIR/models.tag"
echo "=== models.tag = $TAG -- commit it ==="
