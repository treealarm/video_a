#!/usr/bin/env bash
set -euo pipefail

# Fills models/ with the models listed in MODEL_LICENSES.md, from their original publishers, and
# checks every file against the checksum recorded here -- so what is in models/ (and therefore in
# a published treealarm/analytics-models image) can be traced to its source byte for byte. Run it,
# then scripts/push-models-image.sh to publish.
#
# Three of the four are downloaded as published. The primary detector is the exception: it is
# exported here from Ultralytics' yolo26s.pt (checked against its checksum) with pinned versions
# of Ultralytics and OpenVINO; the export is not guaranteed to be byte-identical across machines,
# so its own checksum is printed, not checked. The export runs in a throwaway virtualenv, or with
# EXPORT_PYTHON=/path/to/python, an interpreter that already has the pinned versions.
#
# Needs curl, sha256sum/sha384sum and python3 (with venv).

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${1:-$REPO_DIR/models}"

OMZ=https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1
PLATE_REPO=https://huggingface.co/ishaqinu/yolo26n-license-plate-detection/resolve/b48cca5eabf15fe101e345176c87bcf7c8625318
YOLO26S_URL=https://huggingface.co/Ultralytics/YOLO26/resolve/070ac3c51435984d992ba92eb3f234daf4e4500d/yolo26s.pt
YOLO26S_SHA256=646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b
ULTRALYTICS_VERSION=8.4.84
OPENVINO_VERSION=2026.2.1

mkdir -p "$MODELS_DIR"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# fetch <url> <file in models/> <sha256|sha384> <checksum>
fetch() {
    local tmp="$work/$(basename "$2")"
    curl -sSfL -o "$tmp" "$1"
    if ! echo "$4  $tmp" | "${3}sum" -c --status; then
        echo "checksum mismatch for $1 -- not installed" >&2
        exit 1
    fi
    mv "$tmp" "$MODELS_DIR/$2"
    echo "  $2"
}

# Files of the models these replace. The worker prefers <name>.onnx over <name>.xml, so a stale
# ONNX left beside a new IR would win silently.
rm -f "$MODELS_DIR"/person_embedder.onnx "$MODELS_DIR"/person_embedder.onnx.data \
      "$MODELS_DIR"/plate_detector.xml "$MODELS_DIR"/plate_detector.bin "$MODELS_DIR"/plate_detector.metadata.yaml \
      "$MODELS_DIR"/primary_detector.onnx

echo "=== Open Model Zoo: face-detection-0205 -> face_detector ==="
fetch "$OMZ/face-detection-0205/FP32/face-detection-0205.xml" face_detector.xml sha384 \
    9b0abde1f8d0ab11dae2d3a2c83adae7cb01b35ce3e3c746028c399c16058da9d4057892b042b70d5fbdef5af2d39d12
fetch "$OMZ/face-detection-0205/FP32/face-detection-0205.bin" face_detector.bin sha384 \
    a21f915e45f0785de1e7dc8ca17582ac4ec503b9b9f3bbb0fddbc4e0dfa9475ad82eb453cd10fc27d295cd6755ae2626

echo "=== Open Model Zoo: person-reidentification-retail-0277 -> person_embedder ==="
fetch "$OMZ/person-reidentification-retail-0277/FP32/person-reidentification-retail-0277.xml" person_embedder.xml sha384 \
    8b9bd45ee44c67f08b88c82c33cff1bb0f9e7e687c52741461577badbdb7cde146e49fffedb1f813230cd47420af3470
fetch "$OMZ/person-reidentification-retail-0277/FP32/person-reidentification-retail-0277.bin" person_embedder.bin sha384 \
    adf16b7292f3c3858bd642eed8e471072499872592e242ca2be75fd50100179f72a875b05c1907f56d3d970d751dcba0

echo "=== Hugging Face: ishaqinu/yolo26n-license-plate-detection -> plate_detector ==="
fetch "$PLATE_REPO/alpr_26n.onnx" plate_detector.onnx sha256 \
    f42b5d166a13ab0802973cd8ee04a08225dd102f98488424cccbfe0e7d4e624a

echo "=== Ultralytics: yolo26s.pt -> primary_detector (exported here) ==="
curl -sSfL -o "$work/yolo26s.pt" "$YOLO26S_URL"
echo "$YOLO26S_SHA256  $work/yolo26s.pt" | sha256sum -c --status || { echo "checksum mismatch for $YOLO26S_URL" >&2; exit 1; }
py="${EXPORT_PYTHON:-}"
if [ -z "$py" ]; then
    python3 -m venv "$work/venv"
    "$work/venv/bin/pip" install -q --index-url https://download.pytorch.org/whl/cpu torch torchvision
    "$work/venv/bin/pip" install -q "ultralytics==$ULTRALYTICS_VERSION" "openvino==$OPENVINO_VERSION"
    py="$work/venv/bin/python"
fi
"$py" -c "
import sys, ultralytics, openvino
got = (ultralytics.__version__, openvino.__version__.split('-')[0])
want = ('$ULTRALYTICS_VERSION', '$OPENVINO_VERSION')
sys.exit(0 if got == want else 'export needs ultralytics/openvino %s, %s has %s' % (want, sys.executable, got))
"
# The one-to-many head (end2end=False): the output is every anchor's box and class scores,
# [1, 84, 8400], and inference/primary_detector.cpp does the NMS. Measured against the end-to-end
# head on our clips it finds as many objects or more at the same number of false ones (see
# MODEL_LICENSES.md). Static 640x640, batch 1, FP32.
(cd "$work" && YOLO_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 "$py" -c '
from ultralytics import YOLO
YOLO("yolo26s.pt").export(format="openvino", imgsz=640, half=False, dynamic=False, batch=1, end2end=False)
')
mv "$work/yolo26s_openvino_model/yolo26s.xml" "$MODELS_DIR/primary_detector.xml"
mv "$work/yolo26s_openvino_model/yolo26s.bin" "$MODELS_DIR/primary_detector.bin"
echo "  primary_detector.xml / .bin"
# What the export came to when models/ was last published, with the versions above (Linux x86-64):
#   primary_detector.xml  50f6a99889d91ecd07b25f5675d31050ae2ee9d5096274a58ee80553fd9992f6
#   primary_detector.bin  a844dea661eedbc4ac0aaeb56e998525fd75508806cb5743ec1ef98d143fbc1c
(cd "$MODELS_DIR" && sha256sum primary_detector.xml primary_detector.bin)

echo "=== models/ ==="
ls -l "$MODELS_DIR"
