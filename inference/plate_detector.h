#pragma once

#include <string>
#include <vector>

#include <openvino/openvino.hpp>

#include "detection.h"

struct decoded_frame;

// License-plate detector. Prefer a single-class Ultralytics YOLO export (OpenVINO IR or ONNX,
// output [1, 5, N] = cx,cy,w,h,score) — that is what ships as models/plate_detector today and
// what works on multi-country rear plates under a top-down camera.
//
// Still accepts the older OMZ vehicle-license-plate-detection-barrier-0106 IR (DetectionOutput
// [1,1,N,7]): Chinese barrier / front-facing only, and on mycarplate-style clips it latches onto
// badges ("Pajero") instead of the real plate. Kept so a deployment that has not replaced the
// weights yet does not crash; results will be poor until the YOLO weights are installed.
//
// Bboxes are always full-frame normalized 0..1. Call with the full decoded frame (not a vehicle
// crop): YOLO letterboxes the whole scene; squashing a wide vehicle crop into 300x300 was what
// made the barrier SSD miss readable plates.
class plate_detector {
public:
  explicit plate_detector(const std::string& model_path);

  std::vector<raw_detection> infer(const decoded_frame& frame);

private:
  struct letterbox_meta {
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
  };

  letterbox_meta preprocess_yolo(const decoded_frame& frame, ov::Tensor& input) const;
  void preprocess_ssd(const decoded_frame& frame, ov::Tensor& input) const;
  std::vector<raw_detection> decode_yolo(const ov::Tensor& output, const letterbox_meta& meta,
    int frame_w, int frame_h) const;
  std::vector<raw_detection> decode_ssd(const ov::Tensor& output) const;

  bool m_model_loaded = false;
  bool m_yolo = false; // false → OMZ SSD DetectionOutput path
  bool m_input_nhwc = false;
  int m_input_width = 640;
  int m_input_height = 640;

  ov::Core m_core;
  ov::CompiledModel m_compiled;
  ov::InferRequest m_request;
};
