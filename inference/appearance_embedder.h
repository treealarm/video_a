#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>

#include "detection.h"

struct decoded_frame;

// Generic L2-normalized appearance embedder used by forensic search spaces (person, face, vehicle,
// plate). The caller chooses model_path and label; missing model keeps the embedder disabled.
class appearance_embedder {
public:
  appearance_embedder(std::string model_label, const std::string& model_path);

  // Returns an L2-normalized embedding for bbox on frame, or an empty vector when disabled or when
  // the crop is degenerate.
  std::vector<float> embed(const decoded_frame& frame, const bbox_t& bbox);

  bool loaded() const noexcept { return m_loaded; }
  int dim() const noexcept { return m_dim; }

private:
  std::string m_label;
  std::atomic<bool> m_loaded{false};
  ov::Core m_core;
  ov::CompiledModel m_compiled;
  ov::InferRequest m_request;
  std::mutex m_infer_mu;
  int m_input_h = 256;
  int m_input_w = 128;
  int m_dim = 0;
  bool m_output_f16 = false;
  // ArcFace (and similar face ReID) expects (x/255-0.5)/0.5 on a square crop; OSNet-style body
  // models use ImageNet mean/std. Chosen from the model label (`face` in the string) at load
  // time — never from input shape, so a non-face 112x112 model is not silently mis-normalized.
  bool m_arcface_norm = false;
};
