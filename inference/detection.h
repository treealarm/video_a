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

// Normalized (0..1) bounding box relative to the full frame for primary/plate results, or
// relative to the parent person crop for raw face_detector output (composed to full-frame in
// the pipeline before emit).
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
  // Always normalized to the FULL frame (0..1), for every kind — face bboxes are composed back
  // from their parent person-crop space before ending up here; plates are already full-frame.
  bbox_t bbox;
  std::chrono::system_clock::time_point detected_at;
  std::optional<std::string> recognized_text;
  std::optional<float> text_confidence;
  // Always set for face/license_plate; for person/vehicle only under attach_debug_crops.
  std::vector<uint8_t> crop_jpeg;
  // L2-normalized appearance embedding for re-identification. Carried on every frame of a live
  // track (empty when no embedder model), but only recomputed on START + interval.
  std::vector<float> embedding;
  // True only on the frame the embedding was freshly recomputed; false on the carried-along repeats.
  bool embedding_is_fresh = false;
};
