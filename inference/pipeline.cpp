#include "pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

#include <unordered_set>

#include "primary_detector.h"
#include "face_detector.h"
#include "plate_detector.h"
#include "plate_ocr.h"
#include "tracker.h"
#include "appearance_embedder.h"
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

// Face detector runs on the parent person crop, so its bbox is normalized relative to that crop.
// Compose it back into full-frame coordinates so every emitted detection is in the same space.
bbox_t to_full_frame(const bbox_t& parent, const bbox_t& child)
{
  return bbox_t{
    .x = parent.x + child.x * parent.width,
    .y = parent.y + child.y * parent.height,
    .width = child.width * parent.width,
    .height = child.height * parent.height,
  };
}

// How much larger than the vehicle box a plate may sit in and still count as that vehicle's.
// Plates are found on the full frame (YOLO); this only associates them to a track. Half a box
// covers bumper overhang when the primary vehicle box is tight on the body.
constexpr float kPlateContextMargin = 0.5f;

// Face/plate detections share a parent track_id (or 0). Quantize the box so two faces on the
// same person, or two unassociated plates, keep separate cache slots while still rate-limiting
// each one the way person/vehicle tracks do.
int64_t bbox_slot(const bbox_t& box)
{
  auto q = [](float v) -> int64_t
  {
    return static_cast<int64_t>(std::clamp(v, 0.0f, 1.0f) * 32.0f);
  };
  return (q(box.x) << 15) | (q(box.y) << 10) | (q(box.width) << 5) | q(box.height);
}

// Grow the box by `margin` of its own size on every side, clamped to the frame. Clamping means a
// vehicle at the edge simply gets less context on that side rather than a box hanging outside the
// picture.
bbox_t expand(const bbox_t& box, float margin)
{
  const float x0 = std::clamp(box.x - box.width * margin * 0.5f, 0.0f, 1.0f);
  const float y0 = std::clamp(box.y - box.height * margin * 0.5f, 0.0f, 1.0f);
  const float x1 = std::clamp(box.x + box.width * (1.0f + margin * 0.5f), 0.0f, 1.0f);
  const float y1 = std::clamp(box.y + box.height * (1.0f + margin * 0.5f), 0.0f, 1.0f);
  return bbox_t{ .x = x0, .y = y0, .width = x1 - x0, .height = y1 - y0 };
}

bool contains_center(const bbox_t& box, const bbox_t& inner)
{
  const float cx = inner.x + inner.width * 0.5f;
  const float cy = inner.y + inner.height * 0.5f;
  return cx >= box.x && cx <= box.x + box.width && cy >= box.y && cy <= box.y + box.height;
}

// Minimum confidence from the motion estimator before the gate trusts the answer. A flat /
// saturated-null result has confidence 0 and must not suppress a quiet scene.
constexpr float kMotionGateMinConfidence = 0.15f;
}

static float motion_gate_threshold_from_env()
{
  const char* v = std::getenv("ANALYTICS_MOTION_GATE_THRESHOLD");
  if (!v) return 0.08f;
  try { return std::clamp(std::stof(v), 0.0f, 1.0f); }
  catch (...) { return 0.08f; }
}

pipeline::pipeline(pipeline_config config, const std::string& model_dir)
  : m_config(std::move(config))
  , m_primary(std::make_unique<primary_detector>(model_dir + "/primary_detector"))
  , m_face(std::make_unique<face_detector>(model_dir + "/face_detector"))
  , m_plate(std::make_unique<plate_detector>(model_dir + "/plate_detector"))
  , m_ocr(std::make_unique<plate_ocr>(model_dir + "/plate_ocr"))
  , m_tracker(std::make_unique<tracker>())
  , m_person_embed(std::make_unique<appearance_embedder>("person_embedder", model_dir + "/person_embedder"))
  , m_face_embed(std::make_unique<appearance_embedder>("face_embedder", model_dir + "/face_embedder"))
  , m_vehicle_embed(std::make_unique<appearance_embedder>("vehicle_embedder", model_dir + "/vehicle_embedder"))
  , m_plate_embed(std::make_unique<appearance_embedder>("plate_embedder", model_dir + "/plate_embedder"))
  , m_motion_gate_threshold(motion_gate_threshold_from_env())
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

  // ── Motion gate ──────────────────────────────────────────────────────────
  // When the camera is panning, YOLO invents boxes on the blurred background. IoU cannot keep a
  // track_id across an 8%+ scene shift either (the box moves more than its own width), so the
  // old "suppress_new but keep matching tracks" idea emptied the track set on the first gated
  // frame and then dropped everything until the pan stopped. The honest answer: skip the whole
  // frame — leave the tracker's previous set untouched, emit nothing, rematch on the next quiet
  // sample. Offline ROI footage loses boxes for the duration of a pan; that is preferable to
  // archiving background false positives.
  const auto curr_grid = grayscale_motion_grid(frame);
  bool scene_moving = false;
  if (!m_prev_motion_grid.empty() && !curr_grid.empty())
  {
    if (const auto motion = estimate_global_motion(m_prev_motion_grid, curr_grid);
        motion && motion->confidence >= kMotionGateMinConfidence)
    {
      const float shift = std::hypot(motion->dx, motion->dy);
      if (motion->saturated || shift > m_motion_gate_threshold)
      {
        scene_moving = true;
        log()->debug(
          "pipeline: motion gate triggered shift={:.3f} saturated={} confidence={:.2f} (threshold={:.3f})",
          shift, motion->saturated, motion->confidence, m_motion_gate_threshold);
      }
    }
  }
  if (!curr_grid.empty())
    m_prev_motion_grid = curr_grid;

  // Gated: do not call update — an empty update would wipe m_tracks; we want them frozen.
  const auto tracked = scene_moving
    ? std::vector<tracked_detection>{}
    : m_tracker->update(filtered);

  std::unordered_set<track_key, track_key_hash> live_embed;
  auto maybe_embed_cached = [&](detection_kind kind, int64_t track_id, const bbox_t& bbox,
                                appearance_embedder* embedder, final_detection& out)
  {
    if (embedder == nullptr || !embedder->loaded())
      return;

    const int64_t slot = (kind == detection_kind::face || kind == detection_kind::license_plate)
      ? bbox_slot(bbox) : 0;
    const auto now = std::chrono::steady_clock::now();
    const track_key key{ .kind = kind, .track_id = track_id, .slot = slot };
    live_embed.insert(key);
    auto& cached = m_track_embed[key];
    const bool due = cached.value.empty() ||
      now - cached.computed_at >= std::chrono::seconds(m_config.reid_embed_interval_sec);
    if (due)
    {
      auto fresh = embedder->embed(frame, bbox);
      if (!fresh.empty())
      {
        cached.value = std::move(fresh);
        cached.computed_at = now;
        out.embedding_is_fresh = true;
      }
    }
    out.embedding = cached.value;
  };

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
          .embedding_is_fresh = false,
        };
        if (m_config.attach_debug_crops)
          out.crop_jpeg = encode_crop_jpeg(crop_region(frame, t.bbox));

        // Recompute a re-id embedding on the track's first frame (START) and then at most once per
        // reid_embed_interval_sec — enough for the downstream matcher to re-identify the object
        // without paying an embed on every sampled frame.
        // The inference is rate-limited per live track, while the last computed vector is emitted
        // on every frame of that track. This keeps short tracks searchable even if one queue item
        // is dropped.
        maybe_embed_cached(detection_kind::person, t.track_id, t.bbox, m_person_embed.get(), out);
        emit(out);
      }

      if (wants(detection_kind::face))
      {
        // Run on the person crop, not the full frame: face-detection-0205 point-samples into a
        // fixed 416×416 with no letterbox, so a small face on a 1080p frame becomes a few pixels
        // and is missed. The crop upscales the face enough for the model; compose coords back.
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
          maybe_embed_cached(detection_kind::face, t.track_id, out.bbox, m_face_embed.get(), out);
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
        maybe_embed_cached(detection_kind::vehicle, t.track_id, t.bbox, m_vehicle_embed.get(), out);
        emit(out);
      }

    }
  }

  // Plates are detected on the full frame (not per-vehicle crops): the YOLO plate model letterboxes
  // the scene, and squashing a wide rear-view vehicle crop into a square was what made the old
  // barrier SSD miss readable plates while latching onto badges. Association to a vehicle track
  // is best-effort — a plate with no owner is still emitted (track_id 0) so ROI encoding can use it.
  if (wants(detection_kind::license_plate))
  {
    for (const auto& p : m_plate->infer(frame))
    {
      if (p.confidence < m_config.min_confidence) continue;

      int64_t track_id = 0;
      float best_area = std::numeric_limits<float>::max();
      for (const auto& t : tracked)
      {
        if (t.kind != detection_kind::vehicle) continue;
        const auto context_box = expand(t.bbox, kPlateContextMargin);
        if (!contains_center(context_box, p.bbox)) continue;
        const float area = t.bbox.width * t.bbox.height;
        if (area < best_area)
        {
          best_area = area;
          track_id = t.track_id;
        }
      }

      const auto plate_crop = crop_region(frame, p.bbox);
      const auto ocr = m_ocr->infer(plate_crop);

      final_detection out{
        .track_id = track_id,
        .kind = detection_kind::license_plate,
        .confidence = p.confidence,
        .bbox = p.bbox,
        .detected_at = frame.captured_at,
        .recognized_text = std::nullopt,
        .text_confidence = std::nullopt,
        .crop_jpeg = encode_crop_jpeg(plate_crop),
        .embedding = {},
      };
      maybe_embed_cached(detection_kind::license_plate, track_id, p.bbox, m_plate_embed.get(), out);
      if (ocr)
      {
        out.recognized_text = ocr->text;
        out.text_confidence = ocr->confidence;
      }
      emit(out);
    }
  }

  // Prune cached embeddings down to the keys used this frame — the IoU tracker drops unmatched
  // tracks immediately, and face/plate slots are not in that set (they share a parent track_id).
  if (!m_track_embed.empty())
  {
    for (auto it = m_track_embed.begin(); it != m_track_embed.end();)
      it = live_embed.contains(it->first) ? std::next(it) : m_track_embed.erase(it);
  }
}
