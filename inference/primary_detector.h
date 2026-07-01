#pragma once

#include <chrono>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>

#include "detection.h"

struct decoded_frame;

// Loads a single-output Ultralytics-style YOLO detector (ONNX or OpenVINO IR, shape
// [1, 4+num_classes, num_boxes]) and decodes the COCO "person" class plus a small group of
// COCO "vehicle" classes (car/motorcycle/bus/truck) into PERSON/VEHICLE raw_detections — see
// project plan A.5 (one model covers both classes).
//
// If no model file is found at <model_path>.onnx / <model_path>.xml, falls back to a disabled
// stub: infer() returns nothing (or, if ANALYTICS_STUB_SYNTHETIC_DETECTIONS=true, periodically
// one synthetic PERSON detection) so the rest of the pipeline stays exercisable without models.
class primary_detector {
public:
  explicit primary_detector(const std::string& model_path);

  std::vector<raw_detection> infer(const decoded_frame& frame);

private:
  struct letterbox_meta {
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
  };

  letterbox_meta preprocess(const decoded_frame& frame, ov::Tensor& input) const;
  std::vector<raw_detection> decode_output(const ov::Tensor& output, const letterbox_meta& meta, int frame_w, int frame_h) const;
  std::vector<raw_detection> synthetic_fallback();

  bool m_model_loaded = false;
  int m_input_width = 640;
  int m_input_height = 640;

  ov::Core m_core;
  ov::CompiledModel m_compiled;
  ov::InferRequest m_request;

  bool m_synthetic_enabled = false;
  std::chrono::steady_clock::time_point m_last_synthetic{ std::chrono::steady_clock::time_point::min() };
};
