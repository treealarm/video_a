#pragma once

#include <mutex>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>

#include "detection.h"

struct decoded_frame;

// Computes an L2-normalized person/body appearance embedding (e.g. OSNet 512-d) from a person
// crop, for downstream cross-camera / cross-gap re-identification. The worker assigns no identity
// itself — it only emits the raw vector; matching it to an object_id is the caller's concern.
//
// If no model file is found at <model_path>.onnx / <model_path>.xml, it runs as a disabled stub:
// loaded() is false and embed() returns an empty vector, so the pipeline stays exercisable and the
// embedding field is simply omitted from emitted detections.
class person_embedder {
public:
  explicit person_embedder(const std::string& model_path);

  // Returns an L2-normalized embedding for the bbox region of frame, or an empty vector when the
  // model is not loaded or the crop is degenerate. bbox is normalized (0..1) to the full frame.
  std::vector<float> embed(const decoded_frame& frame, const bbox_t& bbox);

  bool loaded() const noexcept { return m_loaded; }
  int dim() const noexcept { return m_dim; }

private:
  bool m_loaded = false;
  ov::Core m_core;
  ov::CompiledModel m_compiled;
  ov::InferRequest m_request;
  std::string m_input_name;
  std::string m_output_name;
  std::mutex m_infer_mu;
  int m_input_h = 256;
  int m_input_w = 128;
  int m_dim = 0;
};
