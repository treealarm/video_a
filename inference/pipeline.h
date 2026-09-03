#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "detection.h"
#include "frame_sampler.h"
#include "global_motion.h"

class primary_detector;
class face_detector;
class plate_detector;
class plate_ocr;
class tracker;
class appearance_embedder;

struct pipeline_config {
  std::string watch_id;
  std::vector<detection_kind> classes;
  float min_confidence = 0.5f;
  bool attach_debug_crops = false;
  // How often (seconds) to recompute the re-id embedding of a still-live track. It is computed on
  // the first frame a track is seen (START) and then at most once per interval; the most recent
  // vector is attached to every emitted detection of the track regardless.
  int reid_embed_interval_sec = 15;
};

// Orchestrates the 4 inference stages on every sampled (keyframe-decoded) frame — see project
// plan A.5. license_plate in classes always implies running primary_detector for VEHICLE tracks
// (to associate plates), even if "vehicle" itself isn't separately requested — that's an internal
// pipeline detail, not something the caller needs to configure explicitly. Plates themselves are
// detected on the full frame.
class pipeline {
public:
  pipeline(pipeline_config config, const std::string& model_dir);
  ~pipeline();

  void process_frame(const decoded_frame& frame, const std::function<void(const final_detection&)>& emit);

private:
  bool wants(detection_kind kind) const;
  static decoded_frame crop_region(const decoded_frame& frame, const bbox_t& bbox);

  pipeline_config m_config;

  std::unique_ptr<primary_detector> m_primary;
  std::unique_ptr<face_detector> m_face;
  std::unique_ptr<plate_detector> m_plate;
  std::unique_ptr<plate_ocr> m_ocr;
  std::unique_ptr<tracker> m_tracker;
  std::unique_ptr<appearance_embedder> m_person_embed;
  std::unique_ptr<appearance_embedder> m_face_embed;
  std::unique_ptr<appearance_embedder> m_vehicle_embed;
  std::unique_ptr<appearance_embedder> m_plate_embed;

  struct track_key {
    detection_kind kind = detection_kind::person;
    int64_t track_id = 0;
    // Person/vehicle: always 0 (one cache slot per IoU track). Face/plate detections share the
    // parent track_id (or 0 for an unassociated plate), so this quantized bbox keeps distinct
    // objects from overwriting each other's cached vector.
    int64_t slot = 0;
    bool operator==(const track_key& other) const noexcept
    {
      return kind == other.kind && track_id == other.track_id && slot == other.slot;
    }
  };
  struct track_key_hash {
    size_t operator()(const track_key& key) const noexcept
    {
      // kind fits in the three shifted bits; slot is mixed separately so face/plate boxes
      // that share a track_id do not collide in the map.
      return (static_cast<size_t>(key.track_id) << 3)
        ^ static_cast<size_t>(key.kind)
        ^ (static_cast<size_t>(key.slot) * 2654435761u);
    }
  };

  struct track_embedding {
    std::chrono::steady_clock::time_point computed_at{};
    std::vector<float> value;
  };

  // Most recent re-id embedding per live track, and when it was computed. A track absent from this
  // map has not been embedded yet (its next frame is treated as START). Pruned to the
  // currently-tracked set each frame so it can't leak as track ids churn.
  std::unordered_map<track_key, track_embedding, track_key_hash> m_track_embed;

  // ── Motion gate ──────────────────────────────────────────────────────────
  // When the scene shifts more than this (normalized), the frame is dropped wholesale: YOLO on a
  // slewing/blurred picture invents background boxes, and IoU cannot keep a track_id across an
  // 8%+ pan anyway. The previous track set is left intact so the next quiet frame can rematch.
  // Configurable via ANALYTICS_MOTION_GATE_THRESHOLD.
  float m_motion_gate_threshold = 0.08f;
  // Previous coarse grayscale grid (~14 KB), not a full BGR frame copy.
  std::vector<float> m_prev_motion_grid;
};
