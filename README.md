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
- `ANALYTICS_MODEL_PATH` — directory expected to contain `primary_detector.xml/.bin`,
  `face_detector.xml/.bin`, `plate_detector.xml/.bin`, `plate_ocr.xml/.bin` (OpenVINO IR pairs).
  **Phase 1: models are not yet wired in** — all four inference classes are stubs (see
  `inference/*_detector.h`, `inference/plate_ocr.h`) that always return empty results.
- `ANALYTICS_DEVICE` — reserved for future OpenVINO device selection (`CPU`/`GPU`); validated on
  startup but unused until real model wiring lands.
- `ANALYTICS_CROP_STORAGE_PATH` — where JPEG crops (face/plate) are written; own storage, not
  shared with any consumer.
- `ANALYTICS_STUB_SYNTHETIC_DETECTIONS` — optional, dev/test-only. When `true`, periodically emits
  one synthetic `PERSON` detection so the whole pipeline (RTSP → decode → tracker → gRPC stream)
  is exercisable end-to-end without real models.

## Docker

`docker-compose.yml` runs `analytics-worker` plus a Dapr sidecar (`analytics-worker-dapr`) on the
shared `common_app_network` (external — expected to already exist, created by whichever consumer
stack, e.g. vms_rec, starts first). The application binary itself contains zero Dapr-specific
code; the sidecar transparently proxies gRPC calls to its plain `grpc::Service` — see
`grpc_layer/analytics_service_impl.h`.
