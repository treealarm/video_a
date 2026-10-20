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
  // Names the embedding space: vectors are comparable only when this is equal. The model's own
  // name where its IR records one (conversion_parameters/model_name, as Open Model Zoo IRs do),
  // else the file name plus a hash of its contents. Empty while not loaded.
  const std::string& model_id() const noexcept { return m_model_id; }

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
  std::string m_model_id;

  // What the model expects its input to look like. A wrong choice produces no error, only vectors
  // that match nothing, so it is taken from the model file wherever the file says:
  //   bgr255    BGR, 0..255, as decoded. An IR whose conversion recorded mean_values or
  //             scale_values carries its normalization inside the graph, and with
  //             reverse_input_channels the swap to RGB as well -- Open Model Zoo's
  //             person-reidentification IRs are such, and take BGR.
  //   rgb255    RGB, 0..255: the same, converted without reverse_input_channels.
  //   imagenet  RGB, ImageNet mean/std -- an OSNet-style body model exported without them.
  //   arcface   RGB, (x/255-0.5)/0.5 -- ArcFace-style face models.
  // A file that says nothing falls back to the label (`face` in it -> arcface, else imagenet),
  // never to the input shape, so a non-face 112x112 model is not mis-normalized.
  enum class input_norm { bgr255, rgb255, imagenet, arcface };
  input_norm m_norm = input_norm::imagenet;
};
