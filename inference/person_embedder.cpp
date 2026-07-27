#include "person_embedder.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>

#include <openvino/runtime/properties.hpp>

#include "frame_sampler.h"
#include "logging.h"

namespace {
std::string env_or(const char* name, const std::string& fallback)
{
  const char* v = std::getenv(name);
  return v ? v : fallback;
}
}

person_embedder::person_embedder(const std::string& model_path)
{
  std::string resolved;
  if (std::filesystem::exists(model_path + ".onnx"))
    resolved = model_path + ".onnx";
  else if (std::filesystem::exists(model_path + ".xml"))
    resolved = model_path + ".xml";

  if (resolved.empty())
  {
    log()->warn("person_embedder: no model at '{}.[onnx|xml]' — body re-id disabled", model_path);
    return;
  }

  try
  {
    const std::string device = env_or("ANALYTICS_DEVICE", "CPU");
    auto model = m_core.read_model(resolved);
    ov::AnyMap compile_cfg;
    compile_cfg.emplace(ov::hint::inference_precision.name(), ov::element::f32);
    compile_cfg.emplace(ov::inference_num_threads.name(), 1);
    m_compiled = m_core.compile_model(model, device, compile_cfg);
    m_request = m_compiled.create_infer_request();
    m_input_name = m_compiled.input().get_any_name();
    m_output_name = m_compiled.output().get_any_name();

    const auto in_pshape = m_compiled.input().get_partial_shape();
    if (in_pshape.rank().is_static() && in_pshape.rank().get_length() == 4 &&
        in_pshape[2].is_static() && in_pshape[3].is_static())
    {
      m_input_h = static_cast<int>(in_pshape[2].get_length());
      m_input_w = static_cast<int>(in_pshape[3].get_length());
    }

    // Warmup infer to resolve the output dimensionality (handles dynamic-output exports).
    ov::Tensor warmup(ov::element::f32,
                      {1, 3, static_cast<size_t>(m_input_h), static_cast<size_t>(m_input_w)});
    std::fill_n(warmup.data<float>(), warmup.get_size(), 0.0f);
    m_request.set_tensor(m_input_name, warmup);
    m_request.infer();
    m_dim = static_cast<int>(m_request.get_tensor(m_output_name).get_size());
    if (m_dim <= 0)
      m_dim = 512;

    m_loaded = true;
    log()->info("person_embedder: loaded model='{}' device='{}' input={}x{} dim={}",
      resolved, device, m_input_w, m_input_h, m_dim);
  }
  catch (const std::exception& ex)
  {
    log()->warn("person_embedder: failed to load '{}': {} — body re-id disabled", resolved, ex.what());
    m_loaded = false;
  }
}

std::vector<float> person_embedder::embed(const decoded_frame& frame, const bbox_t& bbox)
{
  if (!m_loaded || frame.bgr.empty())
    return {};

  const int x0 = std::clamp(static_cast<int>(bbox.x * frame.width), 0, std::max(0, frame.width - 1));
  const int y0 = std::clamp(static_cast<int>(bbox.y * frame.height), 0, std::max(0, frame.height - 1));
  const int x1 = std::clamp(static_cast<int>((bbox.x + bbox.width) * frame.width), x0 + 1, frame.width);
  const int y1 = std::clamp(static_cast<int>((bbox.y + bbox.height) * frame.height), y0 + 1, frame.height);
  const int cw = x1 - x0;
  const int ch = y1 - y0;
  if (cw <= 0 || ch <= 0)
    return {};

  const int side_h = m_input_h;
  const int side_w = m_input_w;
  const int stride = frame.width * 3;

  std::lock_guard<std::mutex> lk(m_infer_mu);
  ov::Tensor input(ov::element::f32,
                   {1, 3, static_cast<size_t>(side_h), static_cast<size_t>(side_w)});
  auto* dst = input.data<float>();
  const int plane = side_h * side_w;

  // Nearest-neighbor resize of the crop to the ReID input, BGR->RGB, ImageNet normalize.
  constexpr float mean[3] = {0.485f, 0.456f, 0.406f};
  constexpr float stdv[3] = {0.229f, 0.224f, 0.225f};
  for (int y = 0; y < side_h; ++y)
  {
    const int sy = y0 + std::min(ch - 1, y * ch / side_h);
    for (int x = 0; x < side_w; ++x)
    {
      const int sx = x0 + std::min(cw - 1, x * cw / side_w);
      const uint8_t* px = &frame.bgr[static_cast<size_t>(sy) * stride + static_cast<size_t>(sx) * 3];
      const int idx = y * side_w + x;
      dst[0 * plane + idx] = (px[2] / 255.0f - mean[0]) / stdv[0];
      dst[1 * plane + idx] = (px[1] / 255.0f - mean[1]) / stdv[1];
      dst[2 * plane + idx] = (px[0] / 255.0f - mean[2]) / stdv[2];
    }
  }

  m_request.set_tensor(m_input_name, input);
  m_request.infer();
  const ov::Tensor out = m_request.get_tensor(m_output_name);
  const auto* data = out.data<const float>();
  const size_t n = out.get_size();

  std::vector<float> emb(data, data + n);
  double norm = 0.0;
  for (float v : emb)
    norm += static_cast<double>(v) * static_cast<double>(v);
  norm = std::sqrt(norm);
  if (norm > 0.0)
  {
    for (float& v : emb)
      v = static_cast<float>(v / norm);
  }
  return emb;
}
