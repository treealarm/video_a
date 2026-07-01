#pragma once

#include <chrono>
#include <string>
#include <vector>

#include "detection.h"

struct decoded_frame;

// Phase 1 stub — no OpenVINO wiring yet (real model loading is a later, separate step; see
// project plan A.5/A.7). Real interface shape (one infer() call per sampled frame, returns raw
// detections) matches what an ov::InferRequest-backed implementation would expose, so wiring a
// real model later only touches this class, not its callers.
//
// If ANALYTICS_STUB_SYNTHETIC_DETECTIONS=true, periodically emits one synthetic PERSON detection
// so the rest of the pipeline (tracker, gRPC streaming, VmsAnalytics, RetainEvent, ListDetections)
// is exercisable end-to-end without real models.
class primary_detector {
public:
  explicit primary_detector(const std::string& model_dir);

  std::vector<raw_detection> infer(const decoded_frame& frame);

private:
  bool m_synthetic_enabled;
  std::chrono::steady_clock::time_point m_last_synthetic;
};
