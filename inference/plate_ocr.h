#pragma once

#include <optional>
#include <string>

struct decoded_frame;

struct ocr_result {
  std::string text;
  float confidence = 0.0f;
};

// Phase 1 stub — no OpenVINO/PaddleOCR wiring yet (see project plan A.5/A.7). Generic scene-text
// recognition, deliberately not tied to any single country's plate template/alphabet/length.
class plate_ocr {
public:
  explicit plate_ocr(const std::string& model_dir);

  std::optional<ocr_result> infer(const decoded_frame& plate_crop);
};
