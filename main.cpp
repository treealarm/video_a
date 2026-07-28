#include <charconv>
#include <cstdlib>
#include <memory>
#include <string>
#include <system_error>

#include <grpcpp/grpcpp.h>

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

// std::stoi would throw on a non-numeric value, aborting with no log line at all — the opposite of
// the critical+exit(1) contract every other startup check here follows. The range check matters
// just as much: a 0 or negative interval makes the pipeline's "embedding due" test always true,
// silently turning a once-per-interval ReID inference into one per person per sampled frame.
int require_env_int(const char* name, int min_value, int max_value)
{
  const std::string raw = require_env(name);
  int value = 0;
  const auto* const last = raw.data() + raw.size();
  const auto [parse_end, ec] = std::from_chars(raw.data(), last, value);
  if (ec != std::errc{} || parse_end != last)
  {
    log()->critical("Environment variable '{}' must be an integer, got '{}'", name, raw);
    std::exit(1);
  }
  if (value < min_value || value > max_value)
  {
    log()->critical("Environment variable '{}' must be between {} and {}, got {}",
      name, min_value, max_value, value);
    std::exit(1);
  }
  return value;
}
}

int main()
{
  const auto grpc_port = require_env("ANALYTICS_GRPC_PORT");
  const auto model_dir = require_env("ANALYTICS_MODEL_PATH");
  // Reserved for future OpenVINO device selection (CPU/GPU) — validated now so a missing value
  // is visible immediately, even though stub inference doesn't use it yet.
  require_env("ANALYTICS_DEVICE");
  // How often (seconds) a still-live track recomputes its re-id embedding. The vector itself rides
  // along with every detection of the track; this only bounds the inference.
  const int reid_embed_interval_sec = require_env_int("ANALYTICS_REID_EMBED_INTERVAL_SEC", 1, 86400);

  auto queue = std::make_shared<detection_queue>();

  auto watches = std::make_shared<watch_manager>(
    model_dir,
    reid_embed_interval_sec,
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
