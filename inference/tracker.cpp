#include "tracker.h"

#include <algorithm>

namespace {
float overlap_area(const bbox_t& a, const bbox_t& b)
{
  const float ax2 = a.x + a.width, ay2 = a.y + a.height;
  const float bx2 = b.x + b.width, by2 = b.y + b.height;
  const float ix1 = std::max(a.x, b.x), iy1 = std::max(a.y, b.y);
  const float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
  const float iw = std::max(0.0f, ix2 - ix1), ih = std::max(0.0f, iy2 - iy1);
  return iw * ih;
}
}

float tracker::iou(const bbox_t& a, const bbox_t& b)
{
  const float inter = overlap_area(a, b);
  const float area_a = a.width * a.height;
  const float area_b = b.width * b.height;
  const float uni = area_a + area_b - inter;
  return uni > 0.0f ? inter / uni : 0.0f;
}

std::vector<tracked_detection> tracker::update(const std::vector<raw_detection>& detections)
{
  constexpr float kIouThreshold = 0.3f;

  std::vector<track_state> new_tracks;
  std::vector<tracked_detection> result;
  std::vector<bool> used(m_tracks.size(), false);

  for (const auto& det : detections)
  {
    int best_idx = -1;
    float best_iou = kIouThreshold;
    for (size_t i = 0; i < m_tracks.size(); ++i)
    {
      if (used[i] || m_tracks[i].kind != det.kind) continue;
      const float score = iou(m_tracks[i].bbox, det.bbox);
      if (score > best_iou)
      {
        best_iou = score;
        best_idx = static_cast<int>(i);
      }
    }

    int64_t track_id;
    if (best_idx >= 0)
    {
      track_id = m_tracks[static_cast<size_t>(best_idx)].id;
      used[static_cast<size_t>(best_idx)] = true;
    }
    else
    {
      track_id = m_next_id++;
    }

    new_tracks.push_back({ track_id, det.bbox, det.kind });
    result.push_back({ track_id, det.kind, det.confidence, det.bbox });
  }

  m_tracks = std::move(new_tracks);
  return result;
}
