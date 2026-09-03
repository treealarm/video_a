#include "global_motion.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

// Search radius in grid cells. Sized as ~40% of each axis so a fast PTZ pulse between 1 fps
// samples still lands inside the window, while leaving enough pixels for a meaningful MSE.
constexpr int k_max_shift_x = 32; // 32/80 = 0.40 of frame width
constexpr int k_max_shift_y = 18; // 18/45 = 0.40 of frame height
// Scores within this relative band of each other are a flat landscape (uniform frame, no texture).
constexpr float k_flat_ratio = 0.02f;

float sample_grid(const std::vector<float>& grid, int x, int y)
{
  if (x < 0 || y < 0 || x >= k_motion_grid_w || y >= k_motion_grid_h)
    return 0.f;
  return grid[static_cast<size_t>(y * k_motion_grid_w + x)];
}

float score_shift(const std::vector<float>& before, const std::vector<float>& after, int shift_x, int shift_y)
{
  float sum = 0.f;
  int count = 0;
  // Keep the comparison window inside both grids for every candidate shift.
  for (int y = k_max_shift_y; y < k_motion_grid_h - k_max_shift_y; ++y)
  {
    for (int x = k_max_shift_x; x < k_motion_grid_w - k_max_shift_x; ++x)
    {
      const float a = sample_grid(before, x, y);
      const float b = sample_grid(after, x + shift_x, y + shift_y);
      const float diff = a - b;
      sum += diff * diff;
      ++count;
    }
  }
  return count > 0 ? sum / static_cast<float>(count) : 1e9f;
}

}  // namespace

std::vector<float> grayscale_motion_grid(const decoded_frame& frame)
{
  std::vector<float> grid(static_cast<size_t>(k_motion_grid_w * k_motion_grid_h));
  if (frame.width <= 0 || frame.height <= 0 || frame.bgr.empty())
    return {};

  for (int gy = 0; gy < k_motion_grid_h; ++gy)
  {
    const int sy = (gy * frame.height) / k_motion_grid_h;
    for (int gx = 0; gx < k_motion_grid_w; ++gx)
    {
      const int sx = (gx * frame.width) / k_motion_grid_w;
      const size_t off = static_cast<size_t>((sy * frame.width + sx) * 3);
      if (off + 2 >= frame.bgr.size())
        continue;
      const float b = frame.bgr[off];
      const float g = frame.bgr[off + 1];
      const float r = frame.bgr[off + 2];
      grid[static_cast<size_t>(gy * k_motion_grid_w + gx)] = 0.114f * b + 0.587f * g + 0.299f * r;
    }
  }
  return grid;
}

std::optional<global_motion_result> estimate_global_motion(
  const std::vector<float>& before_grid,
  const std::vector<float>& after_grid)
{
  const size_t expected = static_cast<size_t>(k_motion_grid_w * k_motion_grid_h);
  if (before_grid.size() != expected || after_grid.size() != expected)
    return std::nullopt;

  // Seed with zero so a flat score landscape (covered lens, black startup) stays at the origin
  // instead of latching onto the first corner of the search window.
  int best_x = 0;
  int best_y = 0;
  float best_score = score_shift(before_grid, after_grid, 0, 0);
  float second_best = 1e9f;

  for (int dy = -k_max_shift_y; dy <= k_max_shift_y; ++dy)
  {
    for (int dx = -k_max_shift_x; dx <= k_max_shift_x; ++dx)
    {
      if (dx == 0 && dy == 0)
        continue;
      const float score = score_shift(before_grid, after_grid, dx, dy);
      if (score < best_score)
      {
        second_best = best_score;
        best_score = score;
        best_x = dx;
        best_y = dy;
      }
      else if (score < second_best)
      {
        second_best = score;
      }
    }
  }

  global_motion_result result;
  result.dx = static_cast<float>(best_x) / static_cast<float>(k_motion_grid_w);
  result.dy = static_cast<float>(best_y) / static_cast<float>(k_motion_grid_h);
  result.rotation = 0.f;
  result.saturated =
    std::abs(best_x) == k_max_shift_x || std::abs(best_y) == k_max_shift_y;

  // Flat landscape: every shift scores the same (or nearly). Report zero motion with zero
  // confidence so a gate does not latch on a bogus corner and a calibration RPC does not look
  // like a weak-but-valid match (the old floor of 0.3).
  const float ratio = second_best > 1e-6f ? (second_best - best_score) / second_best : 0.f;
  if (ratio < k_flat_ratio)
  {
    result.dx = 0.f;
    result.dy = 0.f;
    result.confidence = 0.f;
    result.saturated = false;
    return result;
  }

  result.confidence = std::clamp(ratio * 1.5f, 0.f, 1.f);
  return result;
}

std::optional<global_motion_result> estimate_global_motion(
  const decoded_frame& before,
  const decoded_frame& after)
{
  if (before.width <= 0 || after.width <= 0)
    return std::nullopt;
  return estimate_global_motion(grayscale_motion_grid(before), grayscale_motion_grid(after));
}
