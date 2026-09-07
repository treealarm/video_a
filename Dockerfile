# syntax=docker/dockerfile:1.5
#
# Builder reuses ta_vms's ta-deps image for the C++ toolchain + shared vcpkg deps
# (protobuf/grpc/spdlog/ffmpeg/openvino) instead of rebuilding them here — see scripts/build.sh,
# which builds ta-deps from the sibling ta_vms checkout first if it's missing locally.

FROM ta-deps AS builder

WORKDIR /build
COPY . /build

RUN cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build -j1

# vcpkg's own libva travels with the binary, because the distribution's older one wins otherwise
# and the first hardware decode dies on a missing symbol -- lazily, hours after startup. Shipping
# ours is safe in the direction that matters: libva looks up the driver entry point by trying
# __vaDriverInit_<major>_<minor> downwards from its own version, so a newer libva loads the
# distribution's older iHD; the reverse is what does not work.
RUN mkdir -p /runtime-libs && \
    cp $(g++ -print-file-name=libstdc++.so.6) /runtime-libs/ && \
    cp $(g++ -print-file-name=libgcc_s.so.1) /runtime-libs/ && \
    cp -P /vcpkg/installed/x64-linux/lib/libva.so.2* /runtime-libs/ && \
    cp -P /vcpkg/installed/x64-linux/lib/libva-drm.so.2* /runtime-libs/

# =========================
# Runtime
# =========================
# 24.04 rather than the 22.04 the builder uses, matching media_server's runtime for the same
# reason: 22.04's intel-media-va-driver is old enough to be a liability on this silicon, and the
# two services share one iGPU. Going the other way is fine -- glibc only works forwards, and a
# binary built against 22.04's 2.35 runs on 2.39.
FROM ubuntu:24.04

# The VA-API driver. Without it the decoder falls back to the CPU, which is a service that merely
# burns processor time -- and "the device is not passed through", "the group is wrong" and "the
# driver did not load" look identical from outside, which is what vainfo is here to separate.
#
# intel-opencl-icd is the other half of the same chip: OpenVINO reaches the iGPU through OpenCL,
# so ANALYTICS_DEVICE=GPU needs a vendor ICD and not only the plugin. Without it OpenVINO
# enumerates CPU alone and the worker refuses to start, naming what it did find -- which is the
# behaviour worth having, because inference is what this service actually spends its time on.
#
# /dev/dri passthrough is opt-in per host; a host that does not grant it leaves
# ANALYTICS_VIDEO_DECODER=auto decoding in software, and says so in the log at startup. The same
# passthrough is what GPU inference needs, so the two arrive together or not at all.
RUN apt-get update && apt-get install -y --no-install-recommends \
    intel-media-va-driver-non-free \
    libva2 \
    libva-drm2 \
    vainfo \
    intel-opencl-icd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /build/build/analytics-worker /app/analytics-worker
COPY --from=builder /runtime-libs /app
COPY --from=builder /build/models /models

ENV LD_LIBRARY_PATH=/app
ENV ANALYTICS_MODEL_PATH=/models
ENV LIBVA_DRIVER_NAME=iHD
# Where the driver actually is. The libva shipped above is vcpkg's, and its compiled-in search
# path is vcpkg's own build prefix -- a directory that does not exist in this image. Without this,
# va_openDriver() looks in /vcpkg/packages/... and returns -1, and all that is said out loud is
# "failed to initialise VAAPI connection", which reads as absent hardware.
ENV LIBVA_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri

CMD ["./analytics-worker"]
