#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Client for ta_vms' RoiStreamTranscode: frames go in with the regions worth keeping sharp, and
// the service muxes the result into a file of its own.
//
// Plain gRPC to a host:port, deliberately. The service is normally reached through Dapr inside
// ta_vms, but an offline pass over a clip has no sidecar and needs none — this is one process
// talking to one other.
//
// The encoder settings are named by string rather than by the generated enum so that a command
// line can carry them unchanged and this header stays free of the protobuf headers. Unknown names
// are rejected at open() rather than silently mapped to a default: "cpu" spelled wrong must not
// quietly become hardware.

struct roi_kind_quality {
  std::string kind;      // matches roi_box::kind
  int32_t qp_delta = 0;  // negative is sharper
  double pad = 0.0;      // halo as a fraction of the frame; 0 leaves the service's default
};

struct roi_encode_settings {
  std::string target;       // host:port of roi-transcode-svc
  std::string output_path;  // absolute, and inside the service's ROI_ALLOWED_ROOTS
  std::string movflags;

  std::string encoder = "auto";  // auto | qsv | vaapi | nvenc | cpu
  // `same` (the default) keeps whatever the input already is: an archive clip that is H.264
  // must come back H.264, or an HLS playlist stitching it with its neighbours breaks.
  std::string codec = "same";    // h265 | h264 | same
  std::string preset = "balanced";  // balanced | fastest | fast | slow | slowest

  int32_t crf = 0;  // 0 takes the encoder's own default
  // 0 means "match the source's keyframe interval". The encoder's own fallback (two seconds)
  // is what a live stream wants; a file pass that rewrites an archive clip wants the same GOP
  // the neighbouring clips already have.
  int32_t gop = 0;
  int32_t max_regions = 0;

  bool has_background_qp_delta = false;
  int32_t background_qp_delta = 0;

  std::vector<roi_kind_quality> kinds;
};

/// One region of one frame, normalized 0..1 from the top-left corner.
struct roi_box {
  std::string kind;
  float x = 0;
  float y = 0;
  float width = 0;
  float height = 0;
  int64_t track_id = 0;
  float confidence = 0;
};

struct roi_encode_report {
  int64_t frames = 0;
  int64_t packets = 0;
  int64_t bytes = 0;
  int64_t regions_dropped = 0;
};

class roi_stream_client {
public:
  roi_stream_client();
  ~roi_stream_client();
  roi_stream_client(const roi_stream_client&) = delete;
  roi_stream_client& operator=(const roi_stream_client&) = delete;

  /// Opens the stream and sends everything that cannot change once encoding has started.
  /// `pix_fmt` is an AVPixelFormat value; raw planes must arrive tightly packed in it.
  bool open(const roi_encode_settings& settings, int32_t width, int32_t height,
    int32_t time_base_num, int32_t time_base_den, int32_t fps_num, int32_t fps_den,
    int32_t pix_fmt, std::string& error);

  /// One frame with the regions that belong to it. `pts` is in the time base given to open()
  /// and must not go backwards.
  bool send_frame(int64_t pts, const uint8_t* data, size_t size,
    const std::vector<roi_box>& boxes);

  /// The same frame, with the regions as a grayscale mask instead: 0 is background, 255 the
  /// strongest quality on offer, and the values in between are graded. The mask is sampled onto
  /// the encoder's block grid, so it does not have to match the frame's size — and should not,
  /// since a full-resolution one would be most of the bytes on the wire.
  bool send_frame_mask(int64_t pts, const uint8_t* data, size_t size,
    const uint8_t* mask, int32_t mask_width, int32_t mask_height);

  /// Half-closes and waits for the service's summary. Must be called for the output file to be
  /// finalized: the muxer's trailer is written when the stream ends.
  bool finish(roi_encode_report& report, std::string& error);

private:
  struct impl;
  std::unique_ptr<impl> m_impl;
};
