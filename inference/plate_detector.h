#pragma once

#include <string>
#include <vector>

#include <openvino/openvino.hpp>

#include "detection.h"

struct decoded_frame;

// License plate detection on VEHICLE-track crops (see project plan A.5). Model: OMZ
// vehicle-license-plate-detection-barrier-0106 (SSD, input 300x300 BGR 0..255, single
// DetectionOutput [1,1,200,7] = image_id,label,confidence,x_min,y_min,x_max,y_max normalized).
// It finds vehicles (label 1) as well as plates (label 2); only plates are kept, since the caller
// already has the vehicle it cropped. Looks for the rectangular plate region only —
// format/alphabet-agnostic by design, so it works regardless of country plate format. Reading the
// characters is plate_ocr's job, not this one. Falls back to stub mode (always empty) when no model
// file is present at {model_path}.[onnx|xml].
class plate_detector {
public:
  explicit plate_detector(const std::string& model_path);

  // Returned bboxes are normalized 0..1 relative to the vehicle crop.
  std::vector<raw_detection> infer(const decoded_frame& vehicle_crop);

private:
  void preprocess(const decoded_frame& frame, ov::Tensor& input) const;

  bool m_model_loaded = false;
  int m_input_width = 300;
  int m_input_height = 300;
  // The IR published by OMZ carries the original TensorFlow NHWC layout, while an ONNX re-export
  // of the same model is NCHW. Both are accepted; which one this is decided at load time.
  bool m_input_nhwc = false;

  ov::Core m_core;
  ov::CompiledModel m_compiled;
  ov::InferRequest m_request;
};
