cmake_minimum_required(VERSION 3.22)
project(VideoAnalytics CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The vcpkg debug tree carries ~15 GB of static OpenVINO/gRPC archives; linking them drags 22 GB
# of DWARF through a single-threaded ld. Our own translation units stay -g -O0, the third-party
# imported targets resolve to their Release variants. Safe on Linux — libstdc++ has no separate
# debug runtime — as long as _GLIBCXX_DEBUG stays off.
set(CMAKE_MAP_IMPORTED_CONFIG_DEBUG Release "")

find_package(PkgConfig REQUIRED)
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(OpenVINO CONFIG REQUIRED)

pkg_check_modules(FFMPEG REQUIRED
    libavformat
    libavcodec
    libavutil
    libswscale
    libswresample
)

set(ANALYTICS_PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/proto/analytics.proto)
# ta_vms' contract, vendored. Only the client half is generated here — this repository never
# serves it, it drives it for the offline ROI pass.
set(ROI_TRANSCODE_PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/proto/roi_transcode.proto)

set(SOURCES
    main.cpp
    logging.h
    ffmpeg_log.cpp
    ffmpeg_log.h
    watch_manager.cpp
    watch_manager.h
    crop_encoder.cpp
    crop_encoder.h

    grpc_layer/analytics_service_impl.cpp
    grpc_layer/analytics_service_impl.h
    grpc_layer/detection_queue.cpp
    grpc_layer/detection_queue.h

    interfaces/av_deleters.h
    interfaces/media_packet.h
    interfaces/media_sink.h
    interfaces/sink_container.h
    interfaces/reader.h

    reader/file_reader.cpp
    reader/file_reader.h
    reader/rtsp_reader.cpp
    reader/rtsp_reader.h

    roi/roi_file_job.cpp
    roi/roi_file_job.h
    roi/roi_stream_client.cpp
    roi/roi_stream_client.h

    sink_container/sink_container_impl.cpp
    sink_container/sink_container_impl.h

    inference/detection.h
    inference/frame_sampler.cpp
    inference/frame_sampler.h
    inference/inference_worker.cpp
    inference/inference_worker.h
    inference/primary_detector.cpp
    inference/primary_detector.h
    inference/face_detector.cpp
    inference/face_detector.h
    inference/plate_detector.cpp
    inference/plate_detector.h
    inference/plate_ocr.cpp
    inference/plate_ocr.h
    inference/tracker.cpp
    inference/tracker.h
    inference/appearance_embedder.cpp
    inference/appearance_embedder.h
    inference/pipeline.cpp
    inference/pipeline.h
)

add_executable(analytics-worker ${SOURCES})

protobuf_generate(
    TARGET analytics-worker
    IMPORT_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/proto
    PROTOS ${ANALYTICS_PROTO_FILE}
)

protobuf_generate(
    TARGET analytics-worker
    LANGUAGE grpc
    IMPORT_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/proto
    PROTOS ${ANALYTICS_PROTO_FILE}
    PLUGIN protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
    GENERATE_EXTENSIONS .grpc.pb.cc .grpc.pb.h
)

protobuf_generate(
    TARGET analytics-worker
    IMPORT_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/proto
    PROTOS ${ROI_TRANSCODE_PROTO_FILE}
)

protobuf_generate(
    TARGET analytics-worker
    LANGUAGE grpc
    IMPORT_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/proto
    PROTOS ${ROI_TRANSCODE_PROTO_FILE}
    PLUGIN protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
    GENERATE_EXTENSIONS .grpc.pb.cc .grpc.pb.h
)

target_include_directories(analytics-worker
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
        ${FFMPEG_INCLUDE_DIRS}
)

target_link_directories(analytics-worker PRIVATE ${FFMPEG_LIBRARY_DIRS})

target_link_libraries(analytics-worker
    PRIVATE
    ${FFMPEG_LIBRARIES}
    protobuf::libprotobuf
    gRPC::grpc
    gRPC::grpc++
    spdlog::spdlog
    openvino::runtime
)

# openvino::openvino_intel_cpu_plugin only exists on static-plugin OpenVINO builds (e.g. vcpkg's,
# where it is a STATIC IMPORTED target). A shared OpenVINO — the pip/apt packages — loads the CPU
# plugin at runtime via plugins.xml and never declares the target at all, so referencing it
# unconditionally fails at generate time. Guard on the target instead of on the toolchain.
if(TARGET openvino::openvino_intel_cpu_plugin)
  target_link_libraries(analytics-worker PRIVATE openvino::openvino_intel_cpu_plugin)

  # Static OpenVINO plugins rely on linker pulling object files with registration
  # symbols; --whole-archive prevents silent zero-output inference at runtime (this was the
  # root cause of YOLOv11's C2PSA attention MatMul crashing while YOLOv8's plain conv blocks
  # happened to link fine without it).
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_link_options(analytics-worker PRIVATE
      "LINKER:--whole-archive"
      "$<TARGET_FILE:openvino::openvino_intel_cpu_plugin>"
      "LINKER:--no-whole-archive"
    )
  endif()
else()
  message(STATUS "OpenVINO CPU plugin is shared (loaded at runtime) — skipping --whole-archive link")
endif()

target_compile_definitions(analytics-worker PRIVATE _CRT_SECURE_NO_WARNINGS)

target_compile_options(analytics-worker PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# Keep our own DWARF in .dwo files instead of copying it into the executable at link time.
target_compile_options(analytics-worker PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-gsplit-dwarf>
)
