#include "appearance_embedder.h"

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

appearance_embedder::appearance_embedder(std::string model_label, const std::string& model_path)
  : m_label(std::move(model_label))
{
  std::string resolved;
  if (std::filesystem::exists(model_path + ".onnx"))
    resolved = model_path + ".onnx";
  else if (std::filesystem::exists(model_path + ".xml"))
    resolved = model_path + ".xml";

  if (resolved.empty())
  {
    log()->warn("{}: no model at '{}.[onnx|xml]' — embedding disabled", m_label, model_path);
    return;
  }

  try
  {
    const std::string device = env_or("ANALYTICS_DEVICE", "CPU");
    auto model = m_core.read_model(resolved);
    ov::AnyMap compile_cfg;
    compile_cfg.emplace(ov::hint::inference_precision.name(), ov::element::f32);
    if (device == "CPU")
      compile_cfg.emplace(ov::inference_num_threads.name(), 1);
    m_compiled = m_core.compile_model(model, device, compile_cfg);
    m_request = m_compiled.create_infer_request();

    const auto input_port = m_compiled.input();
    const auto output_port = m_compiled.output();

    if (input_port.get_element_type() != ov::element::f32)
    {
      log()->warn("{}: model '{}' expects {} input, only f32 is supported — embedding disabled",
        m_label, resolved, input_port.get_element_type().get_type_name());
      return;
    }

    const auto in_pshape = input_port.get_partial_shape();
    if (in_pshape.rank().is_static() && in_pshape.rank().get_length() == 4 &&
      in_pshape[2].is_static() && in_pshape[3].is_static())
    {
      m_input_h = static_cast<int>(in_pshape[2].get_length());
      m_input_w = static_cast<int>(in_pshape[3].get_length());
    }

    ov::Tensor warmup(ov::element::f32,
      {1, 3, static_cast<size_t>(m_input_h), static_cast<size_t>(m_input_w)});
    std::fill_n(warmup.data<float>(), warmup.get_size(), 0.0f);
    m_request.set_tensor(input_port, warmup);
    m_request.infer();

    const ov::Tensor warm_out = m_request.get_tensor(output_port);
    const auto out_type = warm_out.get_element_type();
    if (out_type != ov::element::f32 && out_type != ov::element::f16)
    {
      log()->warn("{}: model '{}' has unsupported {} output (expected f32/f16) — embedding disabled",
        m_label, resolved, out_type.get_type_name());
      return;
    }
    m_output_f16 = out_type == ov::element::f16;

    m_dim = static_cast<int>(warm_out.get_size());
    if (m_dim <= 0)
      m_dim = 512;

    // buffalo_l ArcFace and the old face_embedder path both use this scale; ImageNet mean/std
    // would silently produce a useless space for the same weights. Chosen from the label only
    // — a non-face 112x112 model must not be mis-normalized because of its input shape.
    m_arcface_norm = m_label.find("face") != std::string::npos;

    m_loaded = true;
    log()->info("{}: loaded model='{}' device='{}' input={}x{} dim={} out={} norm={}",
      m_label, resolved, device, m_input_w, m_input_h, m_dim, out_type.get_type_name(),
      m_arcface_norm ? "arcface" : "imagenet");
  }
  catch (const std::exception& ex)
  {
    log()->warn("{}: failed to load '{}': {} — embedding disabled", m_label, model_path, ex.what());
    m_loaded = false;
  }
}

std::vector<float> appearance_embedder::embed(const decoded_frame& frame, const bbox_t& bbox)
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

  constexpr float mean[3] = {0.485f, 0.456f, 0.406f};
  constexpr float stdv[3] = {0.229f, 0.224f, 0.225f};
  const float x_ratio = static_cast<float>(cw) / static_cast<float>(side_w);
  const float y_ratio = static_cast<float>(ch) / static_cast<float>(side_h);
  for (int y = 0; y < side_h; ++y)
  {
    const float fy = std::clamp((static_cast<float>(y) + 0.5f) * y_ratio - 0.5f,
      0.0f, static_cast<float>(ch - 1));
    const int y_lo = static_cast<int>(fy);
    const int y_hi = std::min(y_lo + 1, ch - 1);
    const float wy = fy - static_cast<float>(y_lo);
    const uint8_t* row_lo = &frame.bgr[static_cast<size_t>(y0 + y_lo) * stride];
    const uint8_t* row_hi = &frame.bgr[static_cast<size_t>(y0 + y_hi) * stride];

    for (int x = 0; x < side_w; ++x)
    {
      const float fx = std::clamp((static_cast<float>(x) + 0.5f) * x_ratio - 0.5f,
        0.0f, static_cast<float>(cw - 1));
      const int x_lo = static_cast<int>(fx);
      const int x_hi = std::min(x_lo + 1, cw - 1);
      const float wx = fx - static_cast<float>(x_lo);

      const uint8_t* p00 = row_lo + static_cast<size_t>(x0 + x_lo) * 3;
      const uint8_t* p01 = row_lo + static_cast<size_t>(x0 + x_hi) * 3;
      const uint8_t* p10 = row_hi + static_cast<size_t>(x0 + x_lo) * 3;
      const uint8_t* p11 = row_hi + static_cast<size_t>(x0 + x_hi) * 3;

      auto sample = [&](int c)
        {
          const float top = static_cast<float>(p00[c]) + (static_cast<float>(p01[c]) - static_cast<float>(p00[c])) * wx;
          const float bot = static_cast<float>(p10[c]) + (static_cast<float>(p11[c]) - static_cast<float>(p10[c])) * wx;
          return top + (bot - top) * wy;
        };

      const int idx = y * side_w + x;
      // BGR source → RGB channels in NCHW.
      const float r = sample(2) / 255.0f;
      const float g = sample(1) / 255.0f;
      const float b = sample(0) / 255.0f;
      if (m_arcface_norm)
      {
        dst[0 * plane + idx] = (r - 0.5f) / 0.5f;
        dst[1 * plane + idx] = (g - 0.5f) / 0.5f;
        dst[2 * plane + idx] = (b - 0.5f) / 0.5f;
      }
      else
      {
        dst[0 * plane + idx] = (r - mean[0]) / stdv[0];
        dst[1 * plane + idx] = (g - mean[1]) / stdv[1];
        dst[2 * plane + idx] = (b - mean[2]) / stdv[2];
      }
    }
  }

  std::vector<float> emb;
  try
  {
    m_request.set_tensor(m_compiled.input(), input);
    m_request.infer();
    const ov::Tensor out = m_request.get_tensor(m_compiled.output());
    const size_t n = out.get_size();

    emb.resize(n);
    if (m_output_f16)
    {
      const auto* data = out.data<const ov::float16>();
      for (size_t i = 0; i < n; ++i)
        emb[i] = static_cast<float>(data[i]);
    }
    else
    {
      const auto* data = out.data<const float>();
      std::copy_n(data, n, emb.begin());
    }
  }
  catch (const std::exception& ex)
  {
    log()->error("{}: inference failed ({}) — embedding disabled", m_label, ex.what());
    m_loaded = false;
    return {};
  }

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
