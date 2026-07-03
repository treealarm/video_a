#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

enum class detection_kind {
  person,
  face,
  vehicle,
  license_plate,
};

// Normalized (0..1) bounding box, relative to whatever frame it was produced against — full
// frame for primary_detector results, the parent crop's coordinate space for face/plate results.
struct bbox_t {
  float x = 0;
  float y = 0;
  float width = 0;
  float height = 0;
};

struct raw_detection {
  detection_kind kind = detection_kind::person;
  float confidence = 0.0f;
  bbox_t bbox;
};

struct tracked_detection {
  int64_t track_id = 0;
  detection_kind kind = detection_kind::person;
  float confidence = 0.0f;
  bbox_t bbox;
};

struct final_detection {
  int64_t track_id = 0;
  detection_kind kind = detection_kind::person;
  float confidence = 0.0f;
  bbox_t bbox;
  std::chrono::system_clock::time_point detected_at;
  std::optional<std::string> recognized_text;
  std::optional<float> text_confidence;
  // Always set for face/license_plate; for person/vehicle only under attach_debug_crops.
  std::vector<uint8_t> crop_jpeg;
};
