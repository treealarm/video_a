#include "plate_detector.h"
#include "frame_sampler.h"
#include "logging.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>

namespace {
std::string env_or(const char* name, const std::string& fallback)
{
  const char* v = std::getenv(name);
  return v ? v : fallback;
}

float iou(const bbox_t& a, const bbox_t& b)
{
  const float ax2 = a.x + a.width, ay2 = a.y + a.height;
  const float bx2 = b.x + b.width, by2 = b.y + b.height;
  const float ix1 = std::max(a.x, b.x), iy1 = std::max(a.y, b.y);
  const float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
  const float iw = std::max(0.0f, ix2 - ix1), ih = std::max(0.0f, iy2 - iy1);
  const float inter = iw * ih;
  const float uni = a.width * a.height + b.width * b.height - inter;
  return uni > 0.0f ? inter / uni : 0.0f;
}

// Loose pre-NMS cutoff; pipeline applies the user-configured min_confidence.
constexpr float kMinRawConfidence = 0.25f;
constexpr float kNmsIou = 0.45f;

// Plates are wide rectangles. Badges ("Pajero") and chrome trim that the barrier SSD loved
// often land outside this band or fail the pixel-size floor.
constexpr float kMinAspect = 2.0f;
constexpr float kMaxAspect = 6.5f;
constexpr int kMinWidthPx = 40;
constexpr int kMinHeightPx = 12;

bool plausible_plate(const bbox_t& b, int frame_w, int frame_h)
{
  const float w_px = b.width * static_cast<float>(frame_w);
  const float h_px = b.height * static_cast<float>(frame_h);
  if (w_px < static_cast<float>(kMinWidthPx) || h_px < static_cast<float>(kMinHeightPx))
    return false;
  if (h_px <= 0.0f) return false;
  const float ar = w_px / h_px;
  return ar >= kMinAspect && ar <= kMaxAspect;
}
} // namespace

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

    const auto partial_shape = model->input().get_partial_shape();
    if (!partial_shape.is_static())
    {
      const int64_t reshape_h = partial_shape.rank().is_static() && partial_shape.size() == 4
          && partial_shape[2].is_static()
        ? partial_shape[2].get_length() : m_input_height;
      const int64_t reshape_w = partial_shape.rank().is_static() && partial_shape.size() == 4
          && partial_shape[3].is_static()
        ? partial_shape[3].get_length() : m_input_width;
      model->reshape({ ov::PartialShape{ 1, 3, reshape_h, reshape_w } });
    }

    m_compiled = m_core.compile_model(model, device);
    m_request = m_compiled.create_infer_request();

    const auto input_shape = m_compiled.input().get_shape();
    if (input_shape.size() == 4)
    {
      m_input_nhwc = input_shape[3] == 3;
      m_input_height = static_cast<int>(m_input_nhwc ? input_shape[1] : input_shape[2]);
      m_input_width = static_cast<int>(m_input_nhwc ? input_shape[2] : input_shape[3]);
    }

    // YOLO: [1, 4+C, N]. SSD barrier: [1,1,N,7].
    for (const auto& port : m_compiled.outputs())
    {
      const auto s = port.get_shape();
      if (s.size() == 3 && s[1] >= 5) { m_yolo = true; break; }
      if (s.size() == 4 && s[3] == 7) { m_yolo = false; break; }
    }

    m_model_loaded = true;
    log()->info("plate_detector: loaded '{}' device='{}' input={}x{} layout={} backend={}",
      resolved, device, m_input_width, m_input_height,
      m_input_nhwc ? "NHWC" : "NCHW", m_yolo ? "yolo" : "ssd-barrier");
  }
  catch (const std::exception& ex)
  {
    log()->error("plate_detector: failed to load '{}': {}", resolved, ex.what());
  }
}

plate_detector::letterbox_meta plate_detector::preprocess_yolo(
  const decoded_frame& frame, ov::Tensor& input) const
{
  const float scale = std::min(
    static_cast<float>(m_input_width) / static_cast<float>(frame.width),
    static_cast<float>(m_input_height) / static_cast<float>(frame.height));
  const int new_w = static_cast<int>(std::round(static_cast<float>(frame.width) * scale));
  const int new_h = static_cast<int>(std::round(static_cast<float>(frame.height) * scale));
  const int pad_x = (m_input_width - new_w) / 2;
  const int pad_y = (m_input_height - new_h) / 2;

  auto* dst = input.data<float>();
  const size_t plane = static_cast<size_t>(m_input_width) * static_cast<size_t>(m_input_height);
  std::fill(dst, dst + 3 * plane, 114.0f / 255.0f);

  for (int y = 0; y < new_h; ++y)
  {
    const int sy = std::min(frame.height - 1, static_cast<int>(static_cast<float>(y) / scale));
    for (int x = 0; x < new_w; ++x)
    {
      const int sx = std::min(frame.width - 1, static_cast<int>(static_cast<float>(x) / scale));
      const uint8_t* px = &frame.bgr[(static_cast<size_t>(sy) * static_cast<size_t>(frame.width)
        + static_cast<size_t>(sx)) * 3];
      const size_t idx = static_cast<size_t>(pad_y + y) * static_cast<size_t>(m_input_width)
        + static_cast<size_t>(pad_x + x);
      dst[0 * plane + idx] = static_cast<float>(px[2]) / 255.0f;
      dst[1 * plane + idx] = static_cast<float>(px[1]) / 255.0f;
      dst[2 * plane + idx] = static_cast<float>(px[0]) / 255.0f;
    }
  }
  return { scale, pad_x, pad_y };
}

void plate_detector::preprocess_ssd(const decoded_frame& frame, ov::Tensor& input) const
{
  auto* dst = input.data<float>();
  const size_t plane = static_cast<size_t>(m_input_width) * static_cast<size_t>(m_input_height);

  for (int y = 0; y < m_input_height; ++y)
  {
    const int sy = std::min(frame.height - 1, y * frame.height / m_input_height);
    for (int x = 0; x < m_input_width; ++x)
    {
      const int sx = std::min(frame.width - 1, x * frame.width / m_input_width);
      const uint8_t* px = &frame.bgr[(static_cast<size_t>(sy) * static_cast<size_t>(frame.width)
        + static_cast<size_t>(sx)) * 3];
      const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(m_input_width)
        + static_cast<size_t>(x);

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

std::vector<raw_detection> plate_detector::decode_yolo(const ov::Tensor& output,
  const letterbox_meta& meta, int frame_w, int frame_h) const
{
  const auto shape = output.get_shape();
  if (shape.size() != 3) return {};

  const size_t attrs = shape[1];
  const size_t boxes = shape[2];
  if (attrs < 5) return {};

  const size_t num_classes = attrs - 4;
  const auto* data = output.data<const float>();
  auto at = [&](size_t attr, size_t box) { return data[attr * boxes + box]; };

  std::vector<raw_detection> raw;
  for (size_t b = 0; b < boxes; ++b)
  {
    float best = 0.0f;
    for (size_t c = 0; c < num_classes; ++c)
      best = std::max(best, at(4 + c, b));
    if (best < kMinRawConfidence) continue;

    const float cx = at(0, b);
    const float cy = at(1, b);
    const float bw = at(2, b);
    const float bh = at(3, b);
    const float x1 = (cx - bw / 2.0f - static_cast<float>(meta.pad_x)) / meta.scale;
    const float y1 = (cy - bh / 2.0f - static_cast<float>(meta.pad_y)) / meta.scale;
    const float ww = bw / meta.scale;
    const float hh = bh / meta.scale;

    raw_detection d;
    d.kind = detection_kind::license_plate;
    d.confidence = best;
    d.bbox.x = std::clamp(x1 / static_cast<float>(frame_w), 0.0f, 1.0f);
    d.bbox.y = std::clamp(y1 / static_cast<float>(frame_h), 0.0f, 1.0f);
    d.bbox.width = std::clamp(ww / static_cast<float>(frame_w), 0.0f, 1.0f - d.bbox.x);
    d.bbox.height = std::clamp(hh / static_cast<float>(frame_h), 0.0f, 1.0f - d.bbox.y);
    if (!plausible_plate(d.bbox, frame_w, frame_h)) continue;
    raw.push_back(d);
  }

  std::sort(raw.begin(), raw.end(),
    [](const raw_detection& a, const raw_detection& b) { return a.confidence > b.confidence; });

  std::vector<raw_detection> kept;
  for (const auto& cand : raw)
  {
    bool keep = true;
    for (const auto& k : kept)
    {
      if (iou(cand.bbox, k.bbox) > kNmsIou) { keep = false; break; }
    }
    if (keep) kept.push_back(cand);
  }
  return kept;
}

std::vector<raw_detection> plate_detector::decode_ssd(const ov::Tensor& output) const
{
  constexpr float kPlateLabel = 2.0f;
  const auto rows = output.get_shape()[2];
  const auto* data = output.data<const float>();

  std::vector<raw_detection> result;
  for (size_t i = 0; i < rows; ++i)
  {
    const float* row = data + i * 7;
    if (row[0] < 0.0f) break;
    if (row[1] != kPlateLabel) continue;
    if (row[2] < kMinRawConfidence) continue;

    raw_detection d;
    d.kind = detection_kind::license_plate;
    d.confidence = row[2];
    d.bbox.x = std::clamp(row[3], 0.0f, 1.0f);
    d.bbox.y = std::clamp(row[4], 0.0f, 1.0f);
    d.bbox.width = std::clamp(row[5] - d.bbox.x, 0.0f, 1.0f - d.bbox.x);
    d.bbox.height = std::clamp(row[6] - d.bbox.y, 0.0f, 1.0f - d.bbox.y);
    if (d.bbox.width <= 0.0f || d.bbox.height <= 0.0f) continue;
    // Geometry filter applies here too — drops many badge FPs from the barrier model.
    // Frame size is unknown in this path when coords are already normalized to the input
    // crop; skip pixel floors, keep aspect only.
    if (d.bbox.height > 0.0f)
    {
      const float ar = d.bbox.width / d.bbox.height;
      if (ar < kMinAspect || ar > kMaxAspect) continue;
    }
    result.push_back(d);
  }
  return result;
}

std::vector<raw_detection> plate_detector::infer(const decoded_frame& frame)
{
  if (!m_model_loaded) return {};
  if (frame.width <= 1 || frame.height <= 1 || frame.bgr.empty()) return {};

  if (m_yolo)
  {
    ov::Tensor input(ov::element::f32,
      { 1, 3, static_cast<size_t>(m_input_height), static_cast<size_t>(m_input_width) });
    const auto meta = preprocess_yolo(frame, input);
    m_request.set_tensor(m_compiled.input(), input);
    m_request.infer();
    return decode_yolo(m_request.get_tensor(m_compiled.output()), meta, frame.width, frame.height);
  }

  const ov::Shape shape = m_input_nhwc
    ? ov::Shape{ 1, static_cast<size_t>(m_input_height), static_cast<size_t>(m_input_width), 3 }
    : ov::Shape{ 1, 3, static_cast<size_t>(m_input_height), static_cast<size_t>(m_input_width) };
  ov::Tensor input(ov::element::f32, shape);
  preprocess_ssd(frame, input);
  m_request.set_tensor(m_compiled.input(), input);
  m_request.infer();

  for (const auto& port : m_compiled.outputs())
  {
    auto t = m_request.get_tensor(port);
    const auto s = t.get_shape();
    if (s.size() == 4 && s[3] == 7)
      return decode_ssd(t);
  }
  return {};
}
