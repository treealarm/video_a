#include "face_detector.h"
#include "frame_sampler.h"
#include "logging.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>

namespace {
std::string env_or(const char* name, const std::string& fallback)
{
  const char* v = std::getenv(name);
  return v ? v : fallback;
}

// Loose pre-filter; the pipeline applies the real, user-configured threshold. The model has
// integrated NMS, so no suppression is needed here.
constexpr float kMinRawConfidence = 0.3f;
}

face_detector::face_detector(const std::string& model_path)
{
  std::string resolved;
  if (std::filesystem::exists(model_path + ".onnx"))
    resolved = model_path + ".onnx";
  else if (std::filesystem::exists(model_path + ".xml"))
    resolved = model_path + ".xml";

  if (resolved.empty())
  {
    log()->warn("face_detector: stub mode, no model loaded (model_dir='{}')", model_path);
    return;
  }

  try
  {
    const std::string device = env_or("ANALYTICS_DEVICE", "CPU");
    auto model = m_core.read_model(resolved);
    m_compiled = m_core.compile_model(model, device);
    m_request = m_compiled.create_infer_request();

    const auto input_shape = m_compiled.input().get_shape();
    if (input_shape.size() == 4)
    {
      m_input_height = static_cast<int>(input_shape[2]);
      m_input_width = static_cast<int>(input_shape[3]);
    }

    m_model_loaded = true;
    log()->info("face_detector: loaded '{}' device='{}' input={}x{}",
      resolved, device, m_input_width, m_input_height);
  }
  catch (const std::exception& ex)
  {
    log()->error("face_detector: failed to load '{}': {}", resolved, ex.what());
  }
}

void face_detector::preprocess(const decoded_frame& frame, ov::Tensor& input) const
{
  // Plain resize (no letterbox — OMZ demos resize directly), BGR planar, raw 0..255:
  // face-detection-0205 expects BGR with normalization embedded in the IR.
  auto* dst = input.data<float>();
  const size_t plane = static_cast<size_t>(m_input_width) * static_cast<size_t>(m_input_height);

  for (int y = 0; y < m_input_height; ++y)
  {
    const int sy = std::min(frame.height - 1, y * frame.height / m_input_height);
    for (int x = 0; x < m_input_width; ++x)
    {
      const int sx = std::min(frame.width - 1, x * frame.width / m_input_width);
      const uint8_t* px = &frame.bgr[(static_cast<size_t>(sy) * static_cast<size_t>(frame.width) + static_cast<size_t>(sx)) * 3];
      const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(m_input_width) + static_cast<size_t>(x);

      dst[0 * plane + idx] = static_cast<float>(px[0]);
      dst[1 * plane + idx] = static_cast<float>(px[1]);
      dst[2 * plane + idx] = static_cast<float>(px[2]);
    }
  }
}

std::vector<raw_detection> face_detector::infer(const decoded_frame& person_crop)
{
  if (!m_model_loaded)
    return {};
  if (person_crop.width <= 1 || person_crop.height <= 1 || person_crop.bgr.empty())
    return {};

  ov::Tensor input(ov::element::f32, { 1, 3, static_cast<size_t>(m_input_height), static_cast<size_t>(m_input_width) });
  preprocess(person_crop, input);

  m_request.set_tensor(m_compiled.input(), input);
  m_request.infer();

  // Outputs: "boxes" [N,5] (x_min,y_min,x_max,y_max,confidence in input pixels), "labels" [N].
  // Locate by shape rather than name so an ONNX re-export with different names still works.
  ov::Tensor boxes;
  for (const auto& port : m_compiled.outputs())
  {
    auto t = m_request.get_tensor(port);
    const auto shape = t.get_shape();
    if (shape.size() == 2 && shape[1] == 5) { boxes = t; break; }
  }
  if (!boxes)
    return {};

  const auto shape = boxes.get_shape();
  const auto* data = boxes.data<const float>();

  std::vector<raw_detection> result;
  for (size_t i = 0; i < shape[0]; ++i)
  {
    const float x1 = data[i * 5 + 0];
    const float y1 = data[i * 5 + 1];
    const float x2 = data[i * 5 + 2];
    const float y2 = data[i * 5 + 3];
    const float conf = data[i * 5 + 4];
    if (conf < kMinRawConfidence)
      continue; // boxes are sorted by confidence; the padding tail is all zeros

    raw_detection d;
    d.kind = detection_kind::face;
    d.confidence = conf;
    d.bbox.x = std::clamp(x1 / static_cast<float>(m_input_width), 0.0f, 1.0f);
    d.bbox.y = std::clamp(y1 / static_cast<float>(m_input_height), 0.0f, 1.0f);
    d.bbox.width = std::clamp((x2 - x1) / static_cast<float>(m_input_width), 0.0f, 1.0f);
    d.bbox.height = std::clamp((y2 - y1) / static_cast<float>(m_input_height), 0.0f, 1.0f);
    result.push_back(d);
  }
  return result;
}
