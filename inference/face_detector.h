#pragma once

#include <string>
#include <vector>

#include "detection.h"

struct decoded_frame;

// Phase 1 stub — no OpenVINO wiring yet (see project plan A.5/A.7). Runs on PERSON-track crops.
class face_detector {
public:
  explicit face_detector(const std::string& model_dir);

  std::vector<raw_detection> infer(const decoded_frame& person_crop);
};
