#pragma once

#include <string>
#include <vector>

#include <openvino/openvino.hpp>

#include "detection.h"

struct decoded_frame;

// Face detection on PERSON-track crops (see project plan A.5). Model: OMZ face-detection-0205
// (FCOS head with integrated NMS; input 1x3x416x416 BGR 0..255, outputs "boxes" [N,5] =
// x_min,y_min,x_max,y_max,confidence in input pixels + "labels" [N]). Falls back to stub mode
// (always empty) when no model file is present at {model_path}.[onnx|xml].
class face_detector {
public:
  explicit face_detector(const std::string& model_path);

  // Returned bboxes are normalized 0..1 relative to the person crop.
  std::vector<raw_detection> infer(const decoded_frame& person_crop);

private:
  void preprocess(const decoded_frame& frame, ov::Tensor& input) const;

  bool m_model_loaded = false;
  int m_input_width = 416;
  int m_input_height = 416;

  ov::Core m_core;
  ov::CompiledModel m_compiled;
  ov::InferRequest m_request;
};
