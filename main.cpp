#include <cstdlib>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>

#include "crop_writer.h"
#include "grpc_layer/analytics_service_impl.h"
#include "grpc_layer/detection_queue.h"
#include "logging.h"
#include "watch_manager.h"

namespace {
std::string require_env(const char* name)
{
  const char* v = std::getenv(name);
  if (!v || std::string(v).empty())
  {
    log()->critical("Required environment variable '{}' is not set", name);
    std::exit(1);
  }
  return v;
}
}

int main()
{
  const auto grpc_port = require_env("ANALYTICS_GRPC_PORT");
  const auto model_dir = require_env("ANALYTICS_MODEL_PATH");
  const auto crop_storage_path = require_env("ANALYTICS_CROP_STORAGE_PATH");
  // Reserved for future OpenVINO device selection (CPU/GPU) — validated now so a missing value
  // is visible immediately, even though stub inference doesn't use it yet.
  require_env("ANALYTICS_DEVICE");

  auto queue = std::make_shared<detection_queue>();
  auto crops = std::make_shared<crop_writer>(crop_storage_path);

  auto watches = std::make_shared<watch_manager>(
    model_dir,
    crops,
    [queue](const std::string& watch_id, const final_detection& det)
    {
      queue->push(queued_detection{ watch_id, det });
    });

  analytics_service_impl service(watches, queue);

  const std::string server_address = "0.0.0.0:" + grpc_port;
  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  auto server = builder.BuildAndStart();
  if (!server)
  {
    log()->critical("Failed to start gRPC server on {}", server_address);
    return 1;
  }

  log()->info("analytics-worker listening on {}", server_address);
  server->Wait();
  return 0;
}
