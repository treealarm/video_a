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
- `ANALYTICS_MODEL_PATH` — directory expected to contain `primary_detector`, `face_detector`,
  `plate_detector`, `plate_ocr` and `person_embedder`, each as an OpenVINO IR pair (`.xml/.bin`)
  or an `.onnx` (which wins when both are present). `primary_detector` (person/vehicle, YOLO26s;
  either Ultralytics output layout is read, end-to-end or not), `face_detector` (OMZ
  face-detection-0205), `plate_detector` (YOLO26n single-class plate detector) and
  `person_embedder` (body re-id, OMZ person-reidentification-retail-0277, 256-d) are wired in;
  `plate_ocr` is still a stub that returns empty results, so plates are located but not read. A
  missing model file puts that detector into stub mode with a startup warning.
  `scripts/fetch-models.sh` fills the directory from the sources in `MODEL_LICENSES.md`. The log
  line of each model at startup says what was loaded: a detector's input and output shape, an
  embedder's `id` (the `embedding_model` it stamps on every vector), `dim` and input `norm` —
  `bgr255` (or `rgb255`) when the IR normalizes inside its graph, as Open Model Zoo's do, otherwise `imagenet`
  or `arcface`, and where that choice came from.
- `ANALYTICS_DEVICE` — OpenVINO device selection (`CPU`/`GPU`, or a virtual device such as
  `AUTO`/`MULTI:GPU,CPU`). A real device is checked against `ov::Core::get_available_devices()` at
  startup and a missing one is a startup failure, because a detector that cannot reach its device
  is indistinguishable at runtime from a quiet scene. `GPU` needs Intel's OpenCL runtime
  (`intel-opencl-icd`) as well as the plugin — the image carries it.
- `ANALYTICS_VIDEO_DECODER` — where video is decoded: `auto`, `vaapi`, `qsv`, `nvdec` or `cpu`.
  Not optional. `auto` walks the list and ends at the CPU, so it always starts and says in the log
  which device it took and what it passed over; a named backend stands alone and a machine that
  cannot provide it fails to start, which is the point — hardware that quietly became software
  looks from the outside exactly like a service that merely burns CPU. Hardware decoding also
  needs the render node passed into the container, see `docker-compose.gpu.yml.example`.
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
The encoder muxes the result; this side writes a `*.boxes.json` beside it with the regions that
were asked for.

```bash
analytics-worker --roi-file clip.mp4 --roi-grpc 127.0.0.1:50061 \
    --encoder vaapi --codec h264 --crf 30 \
    --background-qp 8 --qp face=-10 --qp license_plate=-12 --pad 0.02
analytics-worker --help    # the rest of the knobs
```

`--detect-only` is the same pass with nothing on the other end: decode, detect, write
`*.boxes.json`, and never open a gRPC connection or an encoder. It needs the models and nothing
else, which is what makes it usable from a machine that has no service running -- roi_transcode's
bench (`integration/bench/serve.py`) drives its own encoder from the sidecar it writes.
`--regions` stays `boxes` there: the mask and the qp_map grid are painted on the way to the wire.

```bash
analytics-worker --roi-file clip.mp4 --detect-only --boxes clip.boxes.json --detect-fps 2
```

`--regions qp_map` paints detections onto the encoder's coding-block QP grid on this side and
sends that map over the stream (and optionally writes `*.qpmap.json`). Use it to exercise the
library's qp_map path without re-running box→importance→rects on the service.

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

The image bakes `models/` in (the Dockerfile's `models` stage, `ANALYTICS_MODEL_PATH` defaults to
`/models`) so a container works with no extra volume or fetch step. `models/` must hold regular
files only: a symlink reaches the build context as a link and lands in the image dangling, which
the worker reports as a missing model. `models/primary_detector.*` (YOLO26s) and
`models/plate_detector.onnx` (YOLO26n) are Ultralytics models, AGPL-3.0 like Ultralytics YOLO itself, which is
why this worker is AGPL too; `MODEL_LICENSES.md` lists every model with its source and licence.
`models/` itself is gitignored; `scripts/fetch-models.sh` rebuilds it from those sources, checking
each download against a recorded checksum and exporting the primary detector with pinned
Ultralytics/OpenVINO versions. Then `scripts/push-models-image.sh` publishes it for CI.

Linking note: `openvino::openvino_intel_cpu_plugin` must be linked with `--whole-archive`
(see `CMakeLists.txt`) — without it, the static CPU plugin silently drops object files for less
common op kernels (e.g. YOLOv11's C2PSA attention `MatMul`), causing a crash at inference time
even though the model loads successfully.

## CI

`.github/workflows/image.yml` builds the image on every pull request and, on `main`, publishes
`treealarm/analytics-worker:latest` and `:sha-<commit>`. Neither of the Dockerfile's outside inputs
can come from a CI checkout, so both are pinned by a tag file and swapped in for the local name
through a buildx named context:

- `ta-deps.tag` → `treealarm/ta-deps:<tag>`, the git tree hash of `ta_vms/ta-deps/` (see
  `ta_vms/scripts/deps-image-tags.sh`). Published by `ta_install/scripts/push-deps-images.sh`; CI
  does not build it (~4 hours). When ta_vms publishes a new one, bump this file in a commit of its
  own.
- `models.tag` → `treealarm/analytics-models:<tag>`, a hash of the contents of `models/`. After
  changing `models/`, run `scripts/push-models-image.sh` (needs `docker login`, or
  `DOCKERHUB_USERNAME` + `DOCKERHUB_TOKEN`): it builds the image from the Dockerfile's `models`
  stage, pushes it and rewrites `models.tag` — commit that on its own.

A tag that is not published fails the run with a message saying which script publishes it. The
run logs in to Docker Hub with the `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` secrets, so pull
requests from forks cannot build.

## License

GNU Affero General Public License v3.0 (`LICENSE`), because the worker loads and runs Ultralytics
YOLO models, which are AGPL-3.0. The models have licences of their own: see `MODEL_LICENSES.md`.
The rest of TreeAlarm VMS is MIT.
