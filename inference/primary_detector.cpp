#include "primary_detector.h"
#include "frame_sampler.h"
#include "logging.h"

#include <cstdlib>
#include <cstring>

namespace {
bool env_flag(const char* name)
{
  const char* v = std::getenv(name);
  return v != nullptr && std::strcmp(v, "true") == 0;
}
}

primary_detector::primary_detector(const std::string& model_dir)
  : m_synthetic_enabled(env_flag("ANALYTICS_STUB_SYNTHETIC_DETECTIONS"))
  , m_last_synthetic(std::chrono::steady_clock::time_point::min())
{
  log()->warn("primary_detector: stub mode, no model loaded (model_dir='{}')", model_dir);
  if (m_synthetic_enabled)
    log()->info("primary_detector: ANALYTICS_STUB_SYNTHETIC_DETECTIONS=true — emitting synthetic PERSON detections");
}

std::vector<raw_detection> primary_detector::infer(const decoded_frame& /*frame*/)
{
  std::vector<raw_detection> result;

  if (!m_synthetic_enabled)
    return result;

  const auto now = std::chrono::steady_clock::now();
  if (now - m_last_synthetic < std::chrono::seconds(5))
    return result;
  m_last_synthetic = now;

  raw_detection d;
  d.kind = detection_kind::person;
  d.confidence = 0.9f;
  d.bbox = { 0.3f, 0.3f, 0.2f, 0.4f };
  result.push_back(d);
  return result;
}
