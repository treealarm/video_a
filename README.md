# video_a

Standalone, dependency-light RTSP video-analytics engine (person / face / vehicle / license-plate
detection). Generic and product-agnostic: it has no knowledge of any particular VMS's camera
model — the caller tells it "watch this RTSP URL with these parameters" via `StartWatch` /
`StopWatch`, and reads back a stream of `DetectionEvent`s via `StreamDetections`, tagged with the
caller-assigned `watch_id`. See `proto/analytics.proto` for the full contract.

Consumed by [ta_vms](../ta_vms)'s `VmsAnalytics` service over Dapr service invocation (see that
repo's plan doc for the full two-repo architecture) — but nothing here depends on ta_vms, Dapr,
or any particular consumer.

## Building

```sh
cmake --preset linux-default
cmake --build build
```

Produces `build/analytics-worker`. Requires `protobuf`, `grpc`, `spdlog`, `ffmpeg` (vcpkg classic
mode, `x64-linux` triplet) — see `CMakeLists.txt`.

## Running

Requires these environment variables (see `.env`):

- `ANALYTICS_GRPC_PORT` — gRPC listen port.
- `ANALYTICS_MODEL_PATH` — directory expected to contain `primary_detector.xml/.bin` (or `.onnx`),
  `face_detector.xml/.bin`, `plate_detector.xml/.bin`, `plate_ocr.xml/.bin`,
  `person_embedder.xml/.bin` (OpenVINO IR pairs). `primary_detector` (person/vehicle,
  YOLOv11-style), `face_detector` (OMZ face-detection-0205), `plate_detector` (OMZ
  vehicle-license-plate-detection-barrier-0106) and `person_embedder` (body re-id, e.g. OSNet
  512-d) are wired in; `plate_ocr` is still a stub that returns empty results, so plates are
  located but not read. A missing model file puts that detector into stub mode with a startup
  warning.
- `ANALYTICS_DEVICE` — OpenVINO device selection (`CPU`/`GPU`).
- `ANALYTICS_REID_EMBED_INTERVAL_SEC` — how often (seconds, `1`..`86400`) a still-live person track
  recomputes its body re-id embedding. The embedding is computed on the track's first frame and
  re-sent with every detection of that track; this only bounds how often it is *recomputed*. Not
  optional — a missing, non-numeric or out-of-range value is a startup failure (`exit(1)`).
- `ANALYTICS_STUB_SYNTHETIC_DETECTIONS` — optional, dev/test-only. When `true`, periodically emits
  one synthetic `PERSON` detection so the whole pipeline (RTSP → decode → tracker → gRPC stream)
  is exercisable end-to-end without real models.

## Watching a file

`WatchRequest.file_path` names a file to read instead of `rtsp_url`. It is read at its own
timestamps and looped, so everything downstream — the sampler's keyframe-interval measurement most
of all — sees the shape of input it sees from a camera, and a short clip stands in for a
continuous source. Everything else about the watch is unchanged: same classes, same
`sample_fps`, same `StreamDetections`.

## Offline ROI pass (`--roi-file`)

A second entry point, selected by the flag and otherwise not reachable: decode a file, detect what
is in it, and push every frame with its regions to ta_vms' ROI encoder over plain gRPC (no Dapr).
The encoder muxes the result; this side writes a `*.boxes.json` beside it that
`roi_transcode/integration/scripts/boxes-to-ass.py` turns into subtitles for a player.

```bash
analytics-worker --roi-file clip.mp4 --roi-grpc 127.0.0.1:50061 \
    --encoder vaapi --codec h264 --crf 30 \
    --background-qp 8 --qp face=-10 --qp license_plate=-12 --pad 0.02
analytics-worker --help    # the rest of the knobs
```

`roi_transcode/integration/scripts/roi-clip.sh` runs this together with the encoder service and a
player, which is usually what you want.

Two things it does differently from a watch. It does **not** go through `frame_sampler`: that
sheds packets under load, which is right for a live source and wrong for a transcode, so this path
decodes inline on the reader's thread and lets the encoder back-pressure it. And detection stays
sampled (`--detect-fps`) while decoding does not — frames between detection passes carry the
previous regions forward, measured against the file's own timeline so a run is reproducible
whatever the machine's speed.

## Docker

The canonical product deployment (this worker together with the whole ta_vms stack, prebuilt
images) lives in the separate `ta_install` repository — build/publish the `analytics-worker`
image from there via `scripts/build-images.sh` / `scripts/push-images.sh`.

The local `docker-compose.yml` here is a dev convenience: it runs `analytics-worker` plus a Dapr
sidecar (`analytics-worker-dapr`) on the shared `common_app_network` (external — expected to
already exist, created by whichever consumer stack, e.g. ta_vms, starts first). The application
binary itself contains zero Dapr-specific code; the sidecar transparently proxies gRPC calls to
its plain `grpc::Service` — see `grpc_layer/analytics_service_impl.h`.

The Dockerfile's builder stage is `FROM ta-deps` — the same base image `ta_vms/media_server`
builds from, with protobuf/grpc/spdlog/ffmpeg/openvino prebuilt (see that repo's
`ta-deps/Dockerfile`), so this repo's own image build doesn't recompile them. That means a bare
`docker compose build` here only works if `ta-deps` already exists locally; use
`scripts/build.sh` instead — it builds `ta-deps` from a sibling `../ta_vms` checkout first if
missing (override the path with `TA_VMS_DIR`).

The image bakes `models/` in (`COPY --from=builder /build/models /models`, `ANALYTICS_MODEL_PATH`
defaults to `/models`) so a container works with no extra volume or fetch step. **Licensing
note:** `models/primary_detector.*` is a YOLOv11n export and Ultralytics YOLOv11 is AGPL-3.0 —
baking it into a distributed/deployed image is a deliberate, not-yet-fully-reconciled call (see
`ta_install/README.md`'s "Analytics models" section). `models/` itself is gitignored; regenerate
`primary_detector.xml/.bin` with `pip install ultralytics && python -c "from ultralytics import
YOLO; YOLO('yolo11n.pt').export(format='openvino', imgsz=640)"`, and fetch the face detector from
OMZ plus a YOLOv8 single-class plate detector (the OMZ barrier SSD is Chinese front-facing only and
mis-fires on badges like "Pajero" under a top-down rear camera):

```sh
OMZ=https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1
for ext in xml bin; do
  curl -sSLo "models/face_detector.$ext" \
    "$OMZ/face-detection-0205/FP32/face-detection-0205.$ext"
done
# Plate: YOLOv8n fine-tuned on plates (Arijit1080 / Licence-Plate-Detection-using-YOLO-V8),
# exported to OpenVINO IR as models/plate_detector.{xml,bin}.
curl -sSLo /tmp/yolov8n-plate.pt \
  "https://raw.githubusercontent.com/Arijit1080/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR/main/best.pt"
python - <<'PY'
from ultralytics import YOLO
m = YOLO("/tmp/yolov8n-plate.pt")
out = m.export(format="openvino", imgsz=640)
print(out)
PY
# Rename the export's best.xml/.bin into models/plate_detector.xml/.bin
```

Linking note: `openvino::openvino_intel_cpu_plugin` must be linked with `--whole-archive`
(see `CMakeLists.txt`) — without it, the static CPU plugin silently drops object files for less
common op kernels (e.g. YOLOv11's C2PSA attention `MatMul`), causing a crash at inference time
even though the model loads successfully.
