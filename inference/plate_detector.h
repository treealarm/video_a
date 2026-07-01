#pragma once

#include <string>
#include <vector>

#include "detection.h"

struct decoded_frame;

// Phase 1 stub — no OpenVINO wiring yet (see project plan A.5/A.7). Runs on VEHICLE-track crops;
// looks for the rectangular plate region only — format/alphabet-agnostic by design, so it works
// regardless of country plate format. Reading the characters is plate_ocr's job, not this one.
class plate_detector {
public:
  explicit plate_detector(const std::string& model_dir);

  std::vector<raw_detection> infer(const decoded_frame& vehicle_crop);
};
