#include "global_motion.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

// Search radius in grid cells. Sized as ~40% of each axis so a fast PTZ pulse between 1 fps
// samples still lands inside the window, while leaving enough pixels for a meaningful MSE.
constexpr int k_max_shift_x = 32; // 32/80 = 0.40 of frame width
constexpr int k_max_shift_y = 18; // 18/45 = 0.40 of frame height
// Scores within this relative band of each other are a flat landscape (uniform frame, no texture).
constexpr float k_flat_ratio = 0.02f;

// Sum of squared differences with each window's own mean removed.
//
// The two frames are taken a moment apart with a camera movement in between, and the camera's
// exposure control answers the new view by lifting or dropping the whole picture a little. Plain
// SSD reads that change in level as a mismatch at every candidate shift alike, which buries the
// real one: on a scene where auto-exposure moved by a tenth, it matched nothing at all — every
// estimate was rejected as flat. Subtracting the mean compares the structure and ignores the
// level, which is what was wanted from the start.
//
// The window stays the worst-case overlap, the same cells for every candidate, and deliberately:
// scoring each shift over whatever region it happens to share instead makes the means of
// differently sized windows compete, and a small window off at the edge of the search wins on
// having less to disagree about.
float score_shift(const std::vector<float>& before, const std::vector<float>& after, int shift_x, int shift_y)
{
  // Keep the comparison window inside both grids for every candidate shift. With x in
  // [k_max_shift_x, w - k_max_shift_x) and the shift bounded by k_max_shift_x, x + shift_x stays
  // in [0, w) — same for y — so every access below is in range without a per-cell bounds test.
  constexpr int x_begin = k_max_shift_x;
  constexpr int x_end = k_motion_grid_w - k_max_shift_x;
  constexpr int y_begin = k_max_shift_y;
  constexpr int y_end = k_motion_grid_h - k_max_shift_y;
  constexpr int count = (x_end - x_begin) * (y_end - y_begin);
  if constexpr (count <= 0)
    return 1e9f;

  // One pass, via sum((a-b)^2) - sum(a-b)^2/n. Removing each window's mean up front needs two
  // passes and a mean over `before` that is identical for every candidate; this identity gives
  // the same mean-removed SSD from a single traversal. Accumulate in double: the subtraction
  // cancels two large sums against each other when the two frames differ mostly in level.
  double sum_d = 0.0;
  double sum_d2 = 0.0;
  for (int y = y_begin; y < y_end; ++y)
  {
    const int b_base = y * k_motion_grid_w;
    const int a_base = (y + shift_y) * k_motion_grid_w + shift_x;
    for (int x = x_begin; x < x_end; ++x)
    {
      const double d = static_cast<double>(before[static_cast<size_t>(b_base + x)])
        - static_cast<double>(after[static_cast<size_t>(a_base + x)]);
      sum_d += d;
      sum_d2 += d * d;
    }
  }

  constexpr double n = count;
  // Clamped at zero: the identity is exact in real arithmetic, and rounding can only push a
  // window whose true score is zero (identical structure) a hair below it.
  return static_cast<float>(std::max(0.0, (sum_d2 - sum_d * sum_d / n) / n));
}

}  // namespace

// Each cell is the average of the pixels it covers, not one pixel picked out of them.
//
// Point sampling looks like an economy and is a trap. A cell of a 1080p frame is 24x24 pixels, so
// taking its top-left pixel throws away 575 of every 576 — and the survivor is whatever detail
// happened to land on that lattice. Two frames offset by anything other than a whole multiple of
// 24 pixels then sample different physical points, and on a detailed scene those values are
// unrelated: the true shift scores no better than any other, the landscape reads as flat, and the
// estimate is rejected. Detail made it worse rather than better, which is the opposite of what a
// matcher should do. Averaging makes each cell a low-pass sample of its block, so a sub-cell
// shift moves every cell a little instead of scrambling it — the search itself is integer-cell
// and reports no fraction, but the integer answer it does report stays stable and trustworthy.
std::vector<float> grayscale_motion_grid(const decoded_frame& frame)
{
  std::vector<float> grid(static_cast<size_t>(k_motion_grid_w * k_motion_grid_h), 0.f);
  if (frame.width <= 0 || frame.height <= 0 || frame.bgr.empty())
    return {};

  // Whole BGR pixels present in the buffer; a truncated frame simply contributes fewer cells.
  const size_t usable_px = frame.bgr.size() / 3;

  for (int gy = 0; gy < k_motion_grid_h; ++gy)
  {
    const int y_from = (gy * frame.height) / k_motion_grid_h;
    const int y_to = std::max(y_from + 1, ((gy + 1) * frame.height) / k_motion_grid_h);
    for (int gx = 0; gx < k_motion_grid_w; ++gx)
    {
      const int x_from = (gx * frame.width) / k_motion_grid_w;
      const int x_to = std::max(x_from + 1, ((gx + 1) * frame.width) / k_motion_grid_w);

      float sum = 0.f;
      int count = 0;
      for (int y = y_from; y < y_to; ++y)
      {
        // Clamp the row against the pixels the buffer actually holds once, instead of testing
        // every pixel of a multi-megapixel frame against bgr.size().
        const size_t row_base = static_cast<size_t>(y) * static_cast<size_t>(frame.width);
        if (row_base + static_cast<size_t>(x_from) >= usable_px)
          break;
        const int x_last = static_cast<int>(
          std::min(static_cast<size_t>(x_to), usable_px - row_base));
        const uint8_t* px = frame.bgr.data() + (row_base + static_cast<size_t>(x_from)) * 3;
        for (int x = x_from; x < x_last; ++x, px += 3)
        {
          sum += 0.114f * px[0] + 0.587f * px[1] + 0.299f * px[2];
          ++count;
        }
      }
      if (count > 0)
        grid[static_cast<size_t>(gy * k_motion_grid_w + gx)] = sum / static_cast<float>(count);
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

  std::vector<float> scores(
    static_cast<size_t>((2 * k_max_shift_x + 1) * (2 * k_max_shift_y + 1)), 0.f);
  const auto at = [](int dx, int dy) {
    return static_cast<size_t>((dy + k_max_shift_y) * (2 * k_max_shift_x + 1) + dx + k_max_shift_x);
  };

  for (int dy = -k_max_shift_y; dy <= k_max_shift_y; ++dy)
  {
    for (int dx = -k_max_shift_x; dx <= k_max_shift_x; ++dx)
    {
      // Every candidate is scored here, the seeded (0, 0) included: reusing best_score for it
      // would store whatever minimum the scan had reached by then, and that value — not the real
      // score at the origin — would go on to masquerade as a sidelobe.
      const float score = score_shift(before_grid, after_grid, dx, dy);
      scores[at(dx, dy)] = score;
      if (score < best_score)
      {
        best_score = score;
        best_x = dx;
        best_y = dy;
      }
    }
  }

  // Peak to sidelobe: the best score anywhere outside a small exclusion radius around the winner.
  // Measuring against the runner-up instead would compare the peak with its own immediate
  // neighbour, which for a true match on a smoothly varying landscape is nearly as good — so a
  // correct answer looked exactly like no answer at all.
  constexpr int k_exclude = 2;
  float second_best = 1e9f;
  for (int dy = -k_max_shift_y; dy <= k_max_shift_y; ++dy)
    for (int dx = -k_max_shift_x; dx <= k_max_shift_x; ++dx)
      if (std::abs(dx - best_x) > k_exclude || std::abs(dy - best_y) > k_exclude)
        second_best = std::min(second_best, scores[at(dx, dy)]);

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
