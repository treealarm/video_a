#pragma once

#include <cstdint>
#include <vector>

#include "detection.h"

// IoU-based greedy-matching tracker, no Kalman/velocity prediction — see project plan A.5: at
// the low sample_fps this pipeline runs at, simple frame-to-frame IoU overlap is sufficient, and
// avoids over-engineering motion prediction the keyframe-only decode cadence can't reliably feed.
// Tracks not matched in a given frame are dropped immediately (no grace period) — acceptable for
// Phase 1's confidence-threshold + downstream debounce, which already tolerate brief gaps.
class tracker {
public:
  std::vector<tracked_detection> update(const std::vector<raw_detection>& detections);

private:
  struct track_state {
    int64_t id = 0;
    bbox_t bbox;
    detection_kind kind = detection_kind::person;
  };

  std::vector<track_state> m_tracks;
  int64_t m_next_id = 1;

  static float iou(const bbox_t& a, const bbox_t& b);
};
