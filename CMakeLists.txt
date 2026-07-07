cmake_minimum_required(VERSION 3.22)
project(VideoAnalytics CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

set(SOURCES
    main.cpp
    logging.h
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

    reader/rtsp_reader.cpp
    reader/rtsp_reader.h

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

target_compile_definitions(analytics-worker PRIVATE _CRT_SECURE_NO_WARNINGS)

target_compile_options(analytics-worker PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)
