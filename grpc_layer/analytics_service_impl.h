#pragma once

#include <memory>
#include <mutex>
#include <string>

#include <grpcpp/grpcpp.h>

#include "analytics.grpc.pb.h"
#include "detection_queue.h"
#include "watch_manager.h"

class appearance_embedder;
class face_detector;
class plate_detector;
class primary_detector;

class analytics_service_impl final : public analytics::AnalyticsService::Service {
public:
  analytics_service_impl(std::shared_ptr<watch_manager> watches, std::shared_ptr<detection_queue> queue, std::string model_dir);
  ~analytics_service_impl() override;

  analytics_service_impl(const analytics_service_impl&) = delete;
  analytics_service_impl& operator=(const analytics_service_impl&) = delete;

  grpc::Status StartWatch(
    grpc::ServerContext* context,
    const analytics::WatchRequest* request,
    analytics::WatchResponse* response) override;

  grpc::Status StopWatch(
    grpc::ServerContext* context,
    const analytics::StopWatchRequest* request,
    analytics::OperationStatus* response) override;

  grpc::Status StreamDetections(
    grpc::ServerContext* context,
    const analytics::StreamDetectionsRequest* request,
    grpc::ServerWriter<analytics::DetectionEvent>* writer) override;

  grpc::Status EmbedImage(
    grpc::ServerContext* context,
    const analytics::EmbedImageRequest* request,
    analytics::EmbedImageResponse* response) override;

  grpc::Status EstimateGlobalMotion(
    grpc::ServerContext* context,
    const analytics::EstimateGlobalMotionRequest* request,
    analytics::EstimateGlobalMotionResponse* response) override;

private:
  std::shared_ptr<watch_manager> m_watches;
  std::shared_ptr<detection_queue> m_queue;
  // Shared by every EmbedImage call. Each detector holds one ov::InferRequest, so infer() must
  // not overlap across gRPC sync-server threads.
  std::mutex m_detect_mu;
  std::unique_ptr<primary_detector> m_primary;
  std::unique_ptr<face_detector> m_face;
  std::unique_ptr<plate_detector> m_plate;
  std::unique_ptr<appearance_embedder> m_person_embed;
  std::unique_ptr<appearance_embedder> m_face_embed;
  std::unique_ptr<appearance_embedder> m_vehicle_embed;
  std::unique_ptr<appearance_embedder> m_plate_embed;
};
