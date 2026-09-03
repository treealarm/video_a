#pragma once

#include <optional>
#include <vector>

#include "frame_sampler.h"

struct global_motion_result {
  float dx = 0.f;         // normalized image-space shift (scene moved right)
  float dy = 0.f;         // normalized (scene moved down)
  float rotation = 0.f;   // radians, positive = clockwise
  float confidence = 0.f; // 0..1; near zero when the match is flat / unusable
  // True when the best shift sits on the search boundary — the real displacement is at least
  // this large, so a motion gate must treat the frame as moving even if |dx|,|dy| are capped.
  bool saturated = false;
};

// Coarse grayscale grid used by the block matcher. Stored by the pipeline between frames so it
// does not have to keep a full BGR copy (~12 MB) just to compare 80×45 floats.
constexpr int k_motion_grid_w = 80;
constexpr int k_motion_grid_h = 45;

std::vector<float> grayscale_motion_grid(const decoded_frame& frame);

// Estimate how the scene shifted between two grids (or two decoded frames).
std::optional<global_motion_result> estimate_global_motion(
  const std::vector<float>& before_grid,
  const std::vector<float>& after_grid);

std::optional<global_motion_result> estimate_global_motion(
  const decoded_frame& before,
  const decoded_frame& after);
