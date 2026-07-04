# syntax=docker/dockerfile:1.5
#
# Builder reuses vms_rec's vms-deps image for the C++ toolchain + shared vcpkg deps
# (protobuf/grpc/spdlog/ffmpeg/openvino) instead of rebuilding them here — see scripts/build.sh,
# which builds vms-deps from the sibling vms_rec checkout first if it's missing locally.

FROM vms-deps AS builder

WORKDIR /build
COPY . /build

RUN cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build -j1

RUN mkdir -p /runtime-libs && \
    cp $(g++ -print-file-name=libstdc++.so.6) /runtime-libs/ && \
    cp $(g++ -print-file-name=libgcc_s.so.1) /runtime-libs/

# =========================
# Runtime
# =========================
FROM ubuntu:22.04

WORKDIR /app

COPY --from=builder /build/build/analytics-worker /app/analytics-worker
COPY --from=builder /runtime-libs /app
COPY --from=builder /build/models /models

ENV LD_LIBRARY_PATH=/app
ENV ANALYTICS_MODEL_PATH=/models

CMD ["./analytics-worker"]
