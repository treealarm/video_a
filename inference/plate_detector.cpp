#include "plate_detector.h"
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

// Loose pre-filter; the pipeline applies the real, user-configured threshold. The SSD head has
// integrated NMS, so no suppression is needed here.
constexpr float kMinRawConfidence = 0.3f;

// DetectionOutput label ids of the model: 1 is the vehicle, 2 is its plate.
constexpr float kPlateLabel = 2.0f;
}

plate_detector::plate_detector(const std::string& model_path)
{
  std::string resolved;
  if (std::filesystem::exists(model_path + ".onnx"))
    resolved = model_path + ".onnx";
  else if (std::filesystem::exists(model_path + ".xml"))
    resolved = model_path + ".xml";

  if (resolved.empty())
  {
    log()->warn("plate_detector: stub mode, no model loaded (model_dir='{}')", model_path);
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
      m_input_nhwc = input_shape[3] == 3;
      m_input_height = static_cast<int>(m_input_nhwc ? input_shape[1] : input_shape[2]);
      m_input_width = static_cast<int>(m_input_nhwc ? input_shape[2] : input_shape[3]);
    }

    m_model_loaded = true;
    log()->info("plate_detector: loaded '{}' device='{}' input={}x{} layout={}",
      resolved, device, m_input_width, m_input_height, m_input_nhwc ? "NHWC" : "NCHW");
  }
  catch (const std::exception& ex)
  {
    log()->error("plate_detector: failed to load '{}': {}", resolved, ex.what());
  }
}

void plate_detector::preprocess(const decoded_frame& frame, ov::Tensor& input) const
{
  // Plain resize (no letterbox — the SSD was trained on squashed input), BGR, raw 0..255: the
  // normalization is embedded in the IR.
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

      if (m_input_nhwc)
      {
        dst[idx * 3 + 0] = static_cast<float>(px[0]);
        dst[idx * 3 + 1] = static_cast<float>(px[1]);
        dst[idx * 3 + 2] = static_cast<float>(px[2]);
      }
      else
      {
        dst[0 * plane + idx] = static_cast<float>(px[0]);
        dst[1 * plane + idx] = static_cast<float>(px[1]);
        dst[2 * plane + idx] = static_cast<float>(px[2]);
      }
    }
  }
}

std::vector<raw_detection> plate_detector::infer(const decoded_frame& vehicle_crop)
{
  if (!m_model_loaded)
    return {};
  if (vehicle_crop.width <= 1 || vehicle_crop.height <= 1 || vehicle_crop.bgr.empty())
    return {};

  const ov::Shape shape = m_input_nhwc
    ? ov::Shape{ 1, static_cast<size_t>(m_input_height), static_cast<size_t>(m_input_width), 3 }
    : ov::Shape{ 1, 3, static_cast<size_t>(m_input_height), static_cast<size_t>(m_input_width) };
  ov::Tensor input(ov::element::f32, shape);
  preprocess(vehicle_crop, input);

  m_request.set_tensor(m_compiled.input(), input);
  m_request.infer();

  // Output: DetectionOutput [1,1,N,7], rows of image_id,label,confidence,x_min,y_min,x_max,y_max
  // with the coordinates already normalized to the input. Located by shape rather than by name so
  // an ONNX re-export with different names still works.
  ov::Tensor detections;
  for (const auto& port : m_compiled.outputs())
  {
    auto t = m_request.get_tensor(port);
    const auto s = t.get_shape();
    if (s.size() == 4 && s[3] == 7) { detections = t; break; }
  }
  if (!detections)
    return {};

  const auto rows = detections.get_shape()[2];
  const auto* data = detections.data<const float>();

  std::vector<raw_detection> result;
  for (size_t i = 0; i < rows; ++i)
  {
    const float* row = data + i * 7;
    if (row[0] < 0.0f)
      break; // rows are sorted by confidence and terminated by a negative image_id
    if (row[1] != kPlateLabel)
      continue;
    if (row[2] < kMinRawConfidence)
      continue;

    raw_detection d;
    d.kind = detection_kind::license_plate;
    d.confidence = row[2];
    d.bbox.x = std::clamp(row[3], 0.0f, 1.0f);
    d.bbox.y = std::clamp(row[4], 0.0f, 1.0f);
    d.bbox.width = std::clamp(row[5] - d.bbox.x, 0.0f, 1.0f - d.bbox.x);
    d.bbox.height = std::clamp(row[6] - d.bbox.y, 0.0f, 1.0f - d.bbox.y);
    if (d.bbox.width <= 0.0f || d.bbox.height <= 0.0f)
      continue;
    result.push_back(d);
  }
  return result;
}
