#pragma once

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <string>

#include "inference/detection.h"

struct queued_detection {
  std::string watch_id;
  final_detection detection;
};

// Bridges watch_manager's detection callback (called from arbitrary reader threads) to the
// single StreamDetections handler. Bounded, drop-oldest on overflow — see project plan §2.3
// "Устойчивость": detections are derived telemetry, not the source-of-truth recording, so a
// brief consumer stall losing the oldest queued events is an accepted trade-off for Phase 1
// (no disk-persisted outbox, unlike media_server's fs_client.cpp).
class detection_queue {
public:
  void push(queued_detection item);
  std::optional<queued_detection> pop_wait(std::chrono::milliseconds timeout);

private:
  static constexpr size_t kMaxQueueSize = 500;

  std::mutex m_mutex;
  std::condition_variable m_cv;
  std::deque<queued_detection> m_queue;
};
