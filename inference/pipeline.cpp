#include "pipeline.h"

#include <algorithm>
#include <cstring>

#include <unordered_set>

#include "primary_detector.h"
#include "face_detector.h"
#include "plate_detector.h"
#include "plate_ocr.h"
#include "tracker.h"
#include "person_embedder.h"
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

// Face/plate detectors run on the parent crop, so their bbox is normalized relative to that crop,
// not the full frame. Compose it back into full-frame normalized coordinates so every emitted
// detection is in the same (full-frame) space — the consumer draws them all on the full frame.
bbox_t to_full_frame(const bbox_t& parent, const bbox_t& child)
{
  return bbox_t{
    .x = parent.x + child.x * parent.width,
    .y = parent.y + child.y * parent.height,
    .width = child.width * parent.width,
    .height = child.height * parent.height,
  };
}
}

pipeline::pipeline(pipeline_config config, const std::string& model_dir)
  : m_config(std::move(config))
  , m_primary(std::make_unique<primary_detector>(model_dir + "/primary_detector"))
  , m_face(std::make_unique<face_detector>(model_dir + "/face_detector"))
  , m_plate(std::make_unique<plate_detector>(model_dir + "/plate_detector"))
  , m_ocr(std::make_unique<plate_ocr>(model_dir + "/plate_ocr"))
  , m_tracker(std::make_unique<tracker>())
  , m_person_embed(std::make_unique<person_embedder>(model_dir + "/person_embedder"))
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
          .embedding = {},
        };
        if (m_config.attach_debug_crops)
          out.crop_jpeg = encode_crop_jpeg(crop_region(frame, t.bbox));

        // Attach a re-id embedding on the track's first frame (START) and then at most once per
        // reid_embed_interval_sec — enough for the downstream matcher to re-identify the object
        // without paying an embed on every sampled frame.
        if (m_person_embed->loaded())
        {
          const auto now = std::chrono::steady_clock::now();
          const auto it = m_last_embed.find(t.track_id);
          const bool due = it == m_last_embed.end() ||
            now - it->second >= std::chrono::seconds(m_config.reid_embed_interval_sec);
          if (due)
          {
            out.embedding = m_person_embed->embed(frame, t.bbox);
            if (!out.embedding.empty())
              m_last_embed[t.track_id] = now;
          }
        }
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
            .bbox = to_full_frame(t.bbox, f.bbox),
            .detected_at = frame.captured_at,
            .recognized_text = std::nullopt,
            .text_confidence = std::nullopt,
            // Face crops are the caller's persistent artifact — always attached.
            .crop_jpeg = encode_crop_jpeg(crop_region(person_crop, f.bbox)),
            .embedding = {},
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
          .embedding = {},
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
            .bbox = to_full_frame(t.bbox, p.bbox),
            .detected_at = frame.captured_at,
            .recognized_text = std::nullopt,
            .text_confidence = std::nullopt,
            // Plate crops are the caller's persistent artifact — always attached.
            .crop_jpeg = encode_crop_jpeg(plate_crop),
            .embedding = {},
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

  // Prune per-track embedding timestamps down to the tracks seen this frame — the IoU tracker drops
  // unmatched tracks immediately, so anything absent here is gone and must not leak.
  if (!m_last_embed.empty())
  {
    std::unordered_set<int64_t> live;
    live.reserve(tracked.size());
    for (const auto& t : tracked)
      live.insert(t.track_id);
    for (auto it = m_last_embed.begin(); it != m_last_embed.end();)
      it = live.contains(it->first) ? std::next(it) : m_last_embed.erase(it);
  }
}
