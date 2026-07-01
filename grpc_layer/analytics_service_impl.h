#pragma once

#include <memory>

#include <grpcpp/grpcpp.h>

#include "analytics.grpc.pb.h"
#include "detection_queue.h"
#include "watch_manager.h"

class analytics_service_impl final : public analytics::AnalyticsService::Service {
public:
  analytics_service_impl(std::shared_ptr<watch_manager> watches, std::shared_ptr<detection_queue> queue);

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

private:
  std::shared_ptr<watch_manager> m_watches;
  std::shared_ptr<detection_queue> m_queue;
};
