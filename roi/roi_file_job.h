#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "inference/detection.h"
#include "roi/roi_stream_client.h"

// One offline pass over a file: decode it, detect what is in it, and hand every frame to the ROI
// encoder with the regions that belong to it.
//
// A batch job rather than a watch, and the difference is not cosmetic. A watch samples — it
// decodes what it can afford and drops the rest, because a live source that is not drained keeps
// arriving anyway. A transcode cannot drop anything: every frame has to reach the encoder or the
// output is not the same clip. So this path decodes inline, runs at whatever speed the encoder
// accepts, and sheds nothing — which is why it does not go through frame_sampler.
//
// Detection stays sampled even though decoding does not: inference is what costs, and running it
// on every frame of a 30 fps clip buys very little. `detect_fps` is measured against the file's
// own timeline rather than the wall clock, so a pass is reproducible regardless of how fast the
// machine gets through it. Frames between detections carry the previous frame's regions forward —
// which is the same thing the encoder's own `hold_sec` does for file jobs, done here because here
// the boxes are known per frame.

/// How the regions reach the encoder. The ТЗ contract offers both, and they are not the same
/// request: boxes are exact rectangles with a per-kind strength each, a mask is one graded
/// surface sampled onto the encoder's block grid. A mask costs no region budget — which is what
/// matters on a driver that accepts eight of them — and loses the box edges in exchange.
enum class roi_region_form {
  boxes,
  mask,
};

struct roi_job_config {
  std::string input_path;
  /// Sidecar written beside the output: the same regions that were sent to the encoder, so the
  /// result can be checked against what it was asked to protect. Empty skips it.
  std::string boxes_json_path;

  roi_encode_settings encode;

  roi_region_form regions = roi_region_form::boxes;
  /// How much smaller than the frame the mask is drawn. The encoder samples it onto a 32-pixel
  /// block grid, so anything finer than that is bytes on the wire for nothing.
  int mask_scale = 8;

  // Detection
  std::string model_dir;
  std::vector<detection_kind> classes;
  float min_confidence = 0.5f;
  /// Detections per second of content. 0 runs inference on every frame.
  double detect_fps = 2.0;
  int reid_embed_interval_sec = 15;
};

/// Runs to completion. Returns false with the reason already logged.
bool run_roi_file_job(const roi_job_config& config);
