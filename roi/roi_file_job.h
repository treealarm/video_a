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

/// How the regions reach the encoder. The ТЗ contract offers boxes and a mask; qp_map is
/// the same block grid the library builds internally, painted here so the encoder does not
/// re-quantize boxes through a region budget. A mask costs no region budget either, but
/// loses per-kind QP (one 0..255 scale); qp_map keeps signed deltas per block.
enum class roi_region_form {
  boxes,
  mask,
  qp_map,
};

struct roi_job_config {
  std::string input_path;
  /// Sidecar written beside the output: the same regions that were sent to the encoder, so the
  /// result can be checked against what it was asked to protect. Empty skips it.
  std::string boxes_json_path;
  /// Optional qp_map sidecar (JSON). Empty skips; when regions==qp_map and this is left empty
  /// the job still sends maps on the wire without writing a file.
  std::string qpmap_json_path;

  roi_encode_settings encode;

  /// Detect and write the sidecar, with no encoder on the other end: nothing is connected to,
  /// no frame is sent, and `encode` is unused except for what the sidecar records about the
  /// input. It exists because the boxes are useful on their own -- roi_transcode's bench drives
  /// its own encoder from them -- and demanding a gRPC service for a pass that only reads is a
  /// dependency on a running server for no result it produces.
  bool detect_only = false;

  roi_region_form regions = roi_region_form::boxes;
  /// How much smaller than the frame the mask is drawn. The encoder samples it onto a 32-pixel
  /// block grid, so anything finer than that is bytes on the wire for nothing.
  int mask_scale = 8;
  /// Coding-block edge for qp_map. 0 picks 16 for H.264 and 32 otherwise.
  int qpmap_block = 0;

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
