#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "detection.h"
#include "frame_sampler.h"

class primary_detector;
class face_detector;
class plate_detector;
class plate_ocr;
class tracker;
class crop_writer;

struct pipeline_config {
  std::string watch_id;
  std::vector<detection_kind> classes;
  float min_confidence = 0.5f;
  bool attach_debug_crops = false;
};

// Orchestrates the 4 inference stages on every sampled (keyframe-decoded) frame — see project
// plan A.5. license_plate in classes always implies running primary_detector for VEHICLE crops,
// even if "vehicle" itself isn't separately requested — that's an internal pipeline detail, not
// something the caller needs to configure explicitly.
class pipeline {
public:
  pipeline(pipeline_config config, const std::string& model_dir, std::shared_ptr<crop_writer> crops);
  ~pipeline();

  void process_frame(const decoded_frame& frame, const std::function<void(const final_detection&)>& emit);

private:
  bool wants(detection_kind kind) const;
  static decoded_frame crop_region(const decoded_frame& frame, const bbox_t& bbox);

  pipeline_config m_config;
  std::shared_ptr<crop_writer> m_crops;

  std::unique_ptr<primary_detector> m_primary;
  std::unique_ptr<face_detector> m_face;
  std::unique_ptr<plate_detector> m_plate;
  std::unique_ptr<plate_ocr> m_ocr;
  std::unique_ptr<tracker> m_tracker;
};
