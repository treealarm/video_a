#include "pipeline.h"

#include <algorithm>
#include <cstring>

#include "primary_detector.h"
#include "face_detector.h"
#include "plate_detector.h"
#include "plate_ocr.h"
#include "tracker.h"
#include "../crop_encoder.h"
#include "../logging.h"

namespace {
const char* kind_name(detection_kind kind)
{
  switch (kind)
  {
    case detection_kind::person: return "person";
    case detection_kind::face: return "face";
    case detection_kind::vehicle: return "vehicle";
    case detection_kind::license_plate: return "license_plate";
  }
  return "unknown";
}
}

pipeline::pipeline(pipeline_config config, const std::string& model_dir)
  : m_config(std::move(config))
  , m_primary(std::make_unique<primary_detector>(model_dir + "/primary_detector"))
  , m_face(std::make_unique<face_detector>(model_dir + "/face_detector"))
  , m_plate(std::make_unique<plate_detector>(model_dir + "/plate_detector"))
  , m_ocr(std::make_unique<plate_ocr>(model_dir + "/plate_ocr"))
  , m_tracker(std::make_unique<tracker>())
{
}

pipeline::~pipeline() = default;

bool pipeline::wants(detection_kind kind) const
{
  return std::find(m_config.classes.begin(), m_config.classes.end(), kind) != m_config.classes.end();
}

decoded_frame pipeline::crop_region(const decoded_frame& frame, const bbox_t& bbox)
{
  int x = static_cast<int>(bbox.x * static_cast<float>(frame.width));
  int y = static_cast<int>(bbox.y * static_cast<float>(frame.height));
  int w = static_cast<int>(bbox.width * static_cast<float>(frame.width));
  int h = static_cast<int>(bbox.height * static_cast<float>(frame.height));

  x = std::clamp(x, 0, std::max(0, frame.width - 1));
  y = std::clamp(y, 0, std::max(0, frame.height - 1));
  w = std::clamp(w, 1, frame.width - x);
  h = std::clamp(h, 1, frame.height - y);

  decoded_frame out;
  out.width = w;
  out.height = h;
  out.captured_at = frame.captured_at;
  out.bgr.resize(static_cast<size_t>(w) * static_cast<size_t>(h) * 3);

  for (int row = 0; row < h; ++row)
  {
    const uint8_t* src = frame.bgr.data() + (static_cast<size_t>(y + row) * static_cast<size_t>(frame.width) + static_cast<size_t>(x)) * 3;
    uint8_t* dst = out.bgr.data() + static_cast<size_t>(row) * static_cast<size_t>(w) * 3;
    std::memcpy(dst, src, static_cast<size_t>(w) * 3);
  }

  return out;
}

void pipeline::process_frame(const decoded_frame& frame, const std::function<void(const final_detection&)>& emit_out)
{
  const auto& watch_id = m_config.watch_id;
  auto emit = [&](const final_detection& det)
  {
    if (det.recognized_text)
      log()->info("pipeline: watch={} track={} kind={} confidence={:.2f} plate=\"{}\"",
        watch_id, det.track_id, kind_name(det.kind), det.confidence, *det.recognized_text);
    else
      log()->info("pipeline: watch={} track={} kind={} confidence={:.2f}",
        watch_id, det.track_id, kind_name(det.kind), det.confidence);
    emit_out(det);
  };

  auto raw = m_primary->infer(frame);

  std::vector<raw_detection> filtered;
  filtered.reserve(raw.size());
  for (auto& d : raw)
  {
    if (d.confidence >= m_config.min_confidence)
      filtered.push_back(d);
  }

  const auto tracked = m_tracker->update(filtered);

  for (const auto& t : tracked)
  {
    if (t.kind == detection_kind::person)
    {
      if (wants(detection_kind::person))
      {
        final_detection out{
          .track_id = t.track_id,
          .kind = detection_kind::person,
          .confidence = t.confidence,
          .bbox = t.bbox,
          .detected_at = frame.captured_at,
          .recognized_text = std::nullopt,
          .text_confidence = std::nullopt,
          .crop_jpeg = {},
        };
        if (m_config.attach_debug_crops)
          out.crop_jpeg = encode_crop_jpeg(crop_region(frame, t.bbox));
        emit(out);
      }

      if (wants(detection_kind::face))
      {
        const auto person_crop = crop_region(frame, t.bbox);
        for (const auto& f : m_face->infer(person_crop))
        {
          if (f.confidence < m_config.min_confidence) continue;
          final_detection out{
            .track_id = t.track_id,
            .kind = detection_kind::face,
            .confidence = f.confidence,
            .bbox = f.bbox,
            .detected_at = frame.captured_at,
            .recognized_text = std::nullopt,
            .text_confidence = std::nullopt,
            // Face crops are the caller's persistent artifact — always attached.
            .crop_jpeg = encode_crop_jpeg(crop_region(person_crop, f.bbox)),
          };
          emit(out);
        }
      }
    }
    else if (t.kind == detection_kind::vehicle)
    {
      if (wants(detection_kind::vehicle))
      {
        final_detection out{
          .track_id = t.track_id,
          .kind = detection_kind::vehicle,
          .confidence = t.confidence,
          .bbox = t.bbox,
          .detected_at = frame.captured_at,
          .recognized_text = std::nullopt,
          .text_confidence = std::nullopt,
          .crop_jpeg = {},
        };
        if (m_config.attach_debug_crops)
          out.crop_jpeg = encode_crop_jpeg(crop_region(frame, t.bbox));
        emit(out);
      }

      if (wants(detection_kind::license_plate))
      {
        const auto vehicle_crop = crop_region(frame, t.bbox);
        for (const auto& p : m_plate->infer(vehicle_crop))
        {
          if (p.confidence < m_config.min_confidence) continue;

          const auto plate_crop = crop_region(vehicle_crop, p.bbox);
          const auto ocr = m_ocr->infer(plate_crop);

          final_detection out{
            .track_id = t.track_id,
            .kind = detection_kind::license_plate,
            .confidence = p.confidence,
            .bbox = p.bbox,
            .detected_at = frame.captured_at,
            .recognized_text = std::nullopt,
            .text_confidence = std::nullopt,
            // Plate crops are the caller's persistent artifact — always attached.
            .crop_jpeg = encode_crop_jpeg(plate_crop),
          };
          if (ocr)
          {
            out.recognized_text = ocr->text;
            out.text_confidence = ocr->confidence;
          }
          emit(out);
        }
      }
    }
  }
}
