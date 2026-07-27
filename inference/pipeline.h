#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "detection.h"
#include "frame_sampler.h"

class primary_detector;
class face_detector;
class plate_detector;
class plate_ocr;
class tracker;
class person_embedder;

struct pipeline_config {
  std::string watch_id;
  std::vector<detection_kind> classes;
  float min_confidence = 0.5f;
  bool attach_debug_crops = false;
  // How often (seconds) to attach a fresh re-id embedding to a still-live track. An embedding is
  // always attached on the first frame a track is seen (START), then at most once per interval.
  int reid_embed_interval_sec = 15;
};

// Orchestrates the 4 inference stages on every sampled (keyframe-decoded) frame — see project
// plan A.5. license_plate in classes always implies running primary_detector for VEHICLE crops,
// even if "vehicle" itself isn't separately requested — that's an internal pipeline detail, not
// something the caller needs to configure explicitly.
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
  std::unique_ptr<person_embedder> m_person_embed;

  // Last time a re-id embedding was attached, per live track. A track absent from this map has
  // not been embedded yet (its next frame is treated as START). Pruned to the currently-tracked
  // set each frame so it can't leak as track ids churn.
  std::unordered_map<int64_t, std::chrono::steady_clock::time_point> m_last_embed;
};
