#include "primary_detector.h"
#include "frame_sampler.h"
#include "logging.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <unordered_set>

namespace {
bool env_flag(const char* name)
{
  const char* v = std::getenv(name);
  return v != nullptr && std::strcmp(v, "true") == 0;
}

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

// COCO class indices used by Ultralytics-trained YOLO exports (v8/v11 share the same
// [1, 4+num_classes, num_boxes] output layout and COCO-80 class ordering).
constexpr size_t kCocoPerson = 0;
const std::unordered_set<size_t> kCocoVehicles = { 2, 3, 5, 7 }; // car, motorcycle, bus, truck

constexpr float kMinRawConfidence = 0.25f; // loose pre-NMS cutoff; pipeline applies the real, user-configured threshold
constexpr float kNmsIou = 0.45f;
}

primary_detector::primary_detector(const std::string& model_path)
  : m_synthetic_enabled(env_flag("ANALYTICS_STUB_SYNTHETIC_DETECTIONS"))
{
  std::string resolved;
  if (std::filesystem::exists(model_path + ".onnx"))
    resolved = model_path + ".onnx";
  else if (std::filesystem::exists(model_path + ".xml"))
    resolved = model_path + ".xml";

  if (resolved.empty())
  {
    log()->warn("primary_detector: no model at '{}.[onnx|xml]' — running in stub mode", model_path);
    return;
  }

  try
  {
    const std::string device = env_or("ANALYTICS_DEVICE", "CPU");
    auto model = m_core.read_model(resolved);

    // Only reshape if the input truly has dynamic dims — forcing a reshape on an
    // already-static-shaped export was observed to strip tensor name metadata on some
    // exports (surfaced as "Attempt to get a name for a Tensor without names"), so it's
    // applied conditionally rather than unconditionally.
    const auto partial_shape = model->input().get_partial_shape();
    if (!partial_shape.is_static())
    {
      const int64_t reshape_h = partial_shape.rank().is_static() && partial_shape.size() == 4 && partial_shape[2].is_static()
        ? partial_shape[2].get_length() : m_input_height;
      const int64_t reshape_w = partial_shape.rank().is_static() && partial_shape.size() == 4 && partial_shape[3].is_static()
        ? partial_shape[3].get_length() : m_input_width;
      model->reshape({ ov::PartialShape{ 1, 3, reshape_h, reshape_w } });
    }

    m_compiled = m_core.compile_model(model, device);
    m_request = m_compiled.create_infer_request();

    const auto input_shape = m_compiled.input().get_shape();
    if (input_shape.size() == 4)
    {
      m_input_height = static_cast<int>(input_shape[2]);
      m_input_width = static_cast<int>(input_shape[3]);
    }

    m_model_loaded = true;
    log()->info("primary_detector: loaded '{}' device='{}' input={}x{}",
      resolved, device, m_input_width, m_input_height);
  }
  catch (const std::exception& ex)
  {
    log()->error("primary_detector: failed to load '{}': {}", resolved, ex.what());
  }
}

std::vector<raw_detection> primary_detector::synthetic_fallback()
{
  if (!m_synthetic_enabled) return {};

  const auto now = std::chrono::steady_clock::now();
  if (now - m_last_synthetic < std::chrono::seconds(5)) return {};
  m_last_synthetic = now;

  raw_detection d;
  d.kind = detection_kind::person;
  d.confidence = 0.9f;
  d.bbox = { 0.3f, 0.3f, 0.2f, 0.4f };
  return { d };
}

primary_detector::letterbox_meta primary_detector::preprocess(const decoded_frame& frame, ov::Tensor& input) const
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
  std::fill(dst, dst + 3 * plane, 114.0f / 255.0f); // grey letterbox padding, Ultralytics convention

  for (int y = 0; y < new_h; ++y)
  {
    const int sy = std::min(frame.height - 1, static_cast<int>(static_cast<float>(y) / scale));
    for (int x = 0; x < new_w; ++x)
    {
      const int sx = std::min(frame.width - 1, static_cast<int>(static_cast<float>(x) / scale));
      const uint8_t* px = &frame.bgr[(static_cast<size_t>(sy) * static_cast<size_t>(frame.width) + static_cast<size_t>(sx)) * 3];
      const size_t idx = static_cast<size_t>(pad_y + y) * static_cast<size_t>(m_input_width) + static_cast<size_t>(pad_x + x);

      // decoded_frame is BGR24 (frame_sampler's sws_scale target); Ultralytics ONNX exports
      // expect RGB, planar, normalized 0..1 — hence the channel swap here.
      dst[0 * plane + idx] = static_cast<float>(px[2]) / 255.0f;
      dst[1 * plane + idx] = static_cast<float>(px[1]) / 255.0f;
      dst[2 * plane + idx] = static_cast<float>(px[0]) / 255.0f;
    }
  }

  return { scale, pad_x, pad_y };
}

std::vector<raw_detection> primary_detector::decode_output(
  const ov::Tensor& output, const letterbox_meta& meta, int frame_w, int frame_h) const
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
    size_t best_cls = 0;
    for (size_t c = 0; c < num_classes; ++c)
    {
      const float score = at(4 + c, b);
      if (score > best) { best = score; best_cls = c; }
    }
    if (best < kMinRawConfidence) continue;

    detection_kind kind;
    if (best_cls == kCocoPerson) kind = detection_kind::person;
    else if (kCocoVehicles.contains(best_cls)) kind = detection_kind::vehicle;
    else continue;

    const float cx = at(0, b);
    const float cy = at(1, b);
    const float bw = at(2, b);
    const float bh = at(3, b);
    const float x1 = (cx - bw / 2.0f - static_cast<float>(meta.pad_x)) / meta.scale;
    const float y1 = (cy - bh / 2.0f - static_cast<float>(meta.pad_y)) / meta.scale;
    const float ww = bw / meta.scale;
    const float hh = bh / meta.scale;

    raw_detection d;
    d.kind = kind;
    d.confidence = best;
    d.bbox.x = std::clamp(x1 / static_cast<float>(frame_w), 0.0f, 1.0f);
    d.bbox.y = std::clamp(y1 / static_cast<float>(frame_h), 0.0f, 1.0f);
    d.bbox.width = std::clamp(ww / static_cast<float>(frame_w), 0.0f, 1.0f);
    d.bbox.height = std::clamp(hh / static_cast<float>(frame_h), 0.0f, 1.0f);
    raw.push_back(d);
  }

  std::sort(raw.begin(), raw.end(), [](const raw_detection& a, const raw_detection& b) { return a.confidence > b.confidence; });

  std::vector<raw_detection> kept;
  kept.reserve(raw.size());
  for (const auto& cand : raw)
  {
    bool keep = true;
    for (const auto& k : kept)
    {
      if (k.kind != cand.kind) continue; // don't suppress a vehicle against a person box or vice versa
      if (iou(cand.bbox, k.bbox) > kNmsIou) { keep = false; break; }
    }
    if (keep) kept.push_back(cand);
  }
  return kept;
}

std::vector<raw_detection> primary_detector::infer(const decoded_frame& frame)
{
  if (!m_model_loaded)
    return synthetic_fallback();

  ov::Tensor input(ov::element::f32, { 1, 3, static_cast<size_t>(m_input_height), static_cast<size_t>(m_input_width) });
  const auto meta = preprocess(frame, input);

  m_request.set_tensor(m_compiled.input(), input);
  m_request.infer();

  const auto output = m_request.get_tensor(m_compiled.output());
  return decode_output(output, meta, frame.width, frame.height);
}
