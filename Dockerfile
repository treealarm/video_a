# syntax=docker/dockerfile:1.5
#
# Single self-contained multi-stage build — video_a is a small, greenfield repo, so (unlike
# vms_rec's media_server) there's no separate "-deps" base image to maintain yet. Revisit that
# split only if rebuild times become a real problem.

# =========================
# Builder
# =========================
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV VCPKG_DEFAULT_BINARY_CACHE=/vcpkg/binary-cache
ENV VCPKG_BUILD_TYPE=release

RUN apt-get update && apt-get install -y \
    software-properties-common curl zip unzip tar git \
    build-essential cmake ninja-build pkg-config nasm \
    && add-apt-repository ppa:ubuntu-toolchain-r/test -y \
    && apt-get update && apt-get install -y gcc-13 g++-13 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/microsoft/vcpkg.git /vcpkg
RUN /vcpkg/bootstrap-vcpkg.sh

# openvino is installed now (even though Phase 1 inference is stubbed and doesn't link it yet)
# so wiring real models later doesn't require re-running this lengthy step.
RUN --mount=type=cache,target=/vcpkg/buildtrees \
    --mount=type=cache,target=/vcpkg/packages \
    --mount=type=cache,target=/vcpkg/downloads \
    --mount=type=cache,target=/vcpkg/binary-cache \
    VCPKG_BUILD_TYPE=release \
    VCPKG_MAX_CONCURRENCY=1 \
    /vcpkg/vcpkg install \
        protobuf \
        grpc \
        spdlog \
        ffmpeg[gpl,x264,x265,opus,vpx,aom,openssl]:x64-linux \
        openvino:x64-linux

WORKDIR /build
COPY . /build

RUN cmake -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release
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

ENV LD_LIBRARY_PATH=/app

CMD ["./analytics-worker"]
