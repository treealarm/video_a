# video_a

Standalone, dependency-light RTSP video-analytics engine (person / face / vehicle / license-plate
detection). Generic and product-agnostic: it has no knowledge of any particular VMS's camera
model — the caller tells it "watch this RTSP URL with these parameters" via `StartWatch` /
`StopWatch`, and reads back a stream of `DetectionEvent`s via `StreamDetections`, tagged with the
caller-assigned `watch_id`. See `proto/analytics.proto` for the full contract.

Consumed by [vms_rec](../vms_rec)'s `VmsAnalytics` service over Dapr service invocation (see that
repo's plan doc for the full two-repo architecture) — but nothing here depends on vms_rec, Dapr,
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
  `face_detector.xml/.bin`, `plate_detector.xml/.bin`, `plate_ocr.xml/.bin` (OpenVINO IR pairs).
  `primary_detector` (person/vehicle, YOLOv8-style) and `face_detector` (OMZ face-detection-0205)
  are wired in; `plate_detector`/`plate_ocr` are still stubs that return empty results. A missing
  model file puts that detector into stub mode with a startup warning.
- `ANALYTICS_DEVICE` — OpenVINO device selection (`CPU`/`GPU`).
- `ANALYTICS_STUB_SYNTHETIC_DETECTIONS` — optional, dev/test-only. When `true`, periodically emits
  one synthetic `PERSON` detection so the whole pipeline (RTSP → decode → tracker → gRPC stream)
  is exercisable end-to-end without real models.

## Docker

The canonical product deployment (this worker together with the whole vms_rec stack, prebuilt
images) lives in the separate `ta_install` repository — build/publish the `analytics-worker`
image from there via `scripts/build-images.sh` / `scripts/push-images.sh`.

The local `docker-compose.yml` here is a dev convenience: it runs `analytics-worker` plus a Dapr
sidecar (`analytics-worker-dapr`) on the shared `common_app_network` (external — expected to
already exist, created by whichever consumer stack, e.g. vms_rec, starts first). The application
binary itself contains zero Dapr-specific code; the sidecar transparently proxies gRPC calls to
its plain `grpc::Service` — see `grpc_layer/analytics_service_impl.h`.

The Dockerfile's builder stage is `FROM vms-deps` — the same base image `vms_rec/media_server`
builds from, with protobuf/grpc/spdlog/ffmpeg/openvino prebuilt (see that repo's
`vms-deps/Dockerfile`), so this repo's own image build doesn't recompile them. That means a bare
`docker compose build` here only works if `vms-deps` already exists locally; use
`scripts/build.sh` instead — it builds `vms-deps` from a sibling `../vms_rec` checkout first if
missing (override the path with `VMS_REC_DIR`).

The image bakes `models/` in (`COPY --from=builder /build/models /models`, `ANALYTICS_MODEL_PATH`
defaults to `/models`) so a container works with no extra volume or fetch step. **Licensing
note:** `models/primary_detector.*` is a YOLOv8n export and Ultralytics YOLOv8 is AGPL-3.0 —
baking it into a distributed/deployed image is a deliberate, not-yet-fully-reconciled call (see
`ta_install/README.md`'s "Analytics models" section). `models/` itself is gitignored; regenerate
`primary_detector.xml/.bin` with `pip install ultralytics && python -c "from ultralytics import
YOLO; YOLO('yolov8n.pt').export(format='openvino', imgsz=640)"` and `face_detector.xml/.bin` from
OMZ face-detection-0205 (Apache-2.0).
