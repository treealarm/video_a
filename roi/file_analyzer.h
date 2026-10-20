#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "interfaces/av_deleters.h"
#include "interfaces/media_sink.h"
#include "roi/roi_stream_client.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}

class pipeline;

// The part of an offline pass over a file that does not care where the frames go: decode every
// packet, run detection on a schedule of the file's own timeline, carry the regions forward
// between passes, and hand each frame on with the regions in force at that moment.
//
// Two consumers share it. The --roi-file job sends each frame to roi-transcode-svc; the
// FrameAnalysis service streams frames and objects to whoever asked. Both need exactly the same
// frames with exactly the same boxes on them, which is why this is one class and not two copies.
//
// Everything runs on the reader's thread, inline, one packet at a time -- see roi_file_job.h for
// why a batch pass wants no queue.

/// Cheap per-frame description of the picture, for whoever has to decide how to encode it.
/// Measured on the luma plane only, on a subsampled grid -- these are scene statistics, not
/// quality metrics, and a 4K frame does not need sixteen million reads to say "busy" or "dark".
struct frame_scene_stats {
  /// Mean luma, 0..255.
  float mean_luma = 0;
  /// ITU-T P.910 spatial information: std-dev of the Sobel magnitude.
  float spatial_info = 0;
  /// ITU-T P.910 temporal information: std-dev of the difference from the previous frame.
  /// 0 on the first frame.
  float temporal_info = 0;
  /// Global scene shift since the previous frame, normalized to the frame size. Only meaningful
  /// when `motion_confidence` is non-zero.
  float motion_dx = 0;
  float motion_dy = 0;
  float motion_confidence = 0;
};

struct analyzed_frame {
  /// The decoder's own picture: valid only for the duration of the callback.
  const AVFrame* frame = nullptr;
  /// 0-based position in decode output order.
  int64_t index = 0;
  /// Presentation timestamp in `time_base`, nudged so it never goes backwards.
  int64_t pts = 0;
  /// The same position in seconds of the file's own timeline.
  double t = 0;
  /// Detection ran on this very frame; otherwise the regions were carried forward.
  bool detected = false;
  const std::vector<roi_box>* boxes = nullptr;
  AVRational time_base{ 1, 1000 };
  AVRational frame_rate{ 25, 1 };
  std::optional<frame_scene_stats> stats;
};

struct file_analyzer_config {
  /// Detections per second of content. 0 runs inference on every frame, a negative value never
  /// runs it (frames only).
  double detect_fps = 2.0;
  bool scene_stats = false;
  /// Detect on the picture turned the way the file's display matrix says to show it (a phone
  /// held sideways or upside down), and give the boxes back in the stored picture's coordinates.
  /// The frames themselves always stay as stored.
  bool upright = false;
};

class file_analyzer final : public media_sink {
public:
  /// Returns false to stop the pass; the analyzer then reports failed().
  using handler = std::function<bool(file_analyzer&, const analyzed_frame&)>;

  /// `detector` may be null only when detection is off (detect_fps < 0).
  file_analyzer(file_analyzer_config config, pipeline* detector, handler on_frame);
  ~file_analyzer() override;

  void on_packet(const std::shared_ptr<media_packet>& pkt) override;

  /// Pushes the last frames out of the decoder. The reader has no way to say "that was the end"
  /// through media_sink, so the job calls this once the read loop has finished.
  void flush();

  /// Asks the pass to stop at the next frame. Safe from any thread.
  void cancel() { m_cancelled = true; }

  bool failed() const { return m_failed; }
  bool cancelled() const { return m_cancelled; }
  int64_t frames() const { return m_frames; }
  int64_t detections_run() const { return m_detect_passes; }

  /// Tightly packed YUV420P of `frame`, converted when the decoder produced anything else.
  /// Lazily, because a consumer that only wants the boxes should not pay for the copy. Returns
  /// null (and marks the pass failed) when the conversion cannot be built.
  const std::vector<uint8_t>* pack_yuv420p(const AVFrame* frame);

private:
  struct sws_deleter {
    void operator()(SwsContext* ctx) const noexcept { if (ctx) sws_freeContext(ctx); }
  };
  using sws_ptr = std::unique_ptr<SwsContext, sws_deleter>;
  using codec_ptr = std::unique_ptr<AVCodecContext, avcodec_context_deleter>;

  bool ensure_decoder(const media_packet& pkt);
  void drain_decoder();
  void handle_frame(const AVFrame* frame);
  double frame_time_sec(const AVFrame* frame) const;
  bool detection_due(double t) const;
  void run_detection(const AVFrame* frame, double t);
  bool ensure_bgr_context(const AVFrame* frame);
  frame_scene_stats measure_scene(const AVFrame* frame);

  file_analyzer_config m_config;
  pipeline* m_detector = nullptr;
  handler m_on_frame;

  codec_ptr m_decoder;
  AVRational m_time_base{ 1, 1000 };
  AVRational m_frame_rate{ 25, 1 };

  // Clockwise degrees a player turns the stored picture by: 0, 90, 180 or 270. Detection uses it
  // only with `upright`.
  int m_rotation = 0;

  sws_ptr m_bgr_ctx;
  int m_bgr_width = 0;
  int m_bgr_height = 0;
  AVPixelFormat m_bgr_format = AV_PIX_FMT_NONE;

  sws_ptr m_yuv_ctx;
  int m_yuv_width = 0;
  int m_yuv_height = 0;
  AVPixelFormat m_yuv_format = AV_PIX_FMT_NONE;
  std::vector<uint8_t> m_yuv;

  // The regions in force. Carried forward between detection passes, which is what makes a clip
  // detected at 2 Hz encodable at 30 fps without the boxes flickering in and out.
  std::vector<roi_box> m_held;
  double m_next_detect_at = 0;
  int64_t m_detect_passes = 0;

  // Scene statistics keep the previous frame's subsampled luma and motion grid.
  std::vector<uint8_t> m_prev_luma;
  std::vector<float> m_prev_grid;

  std::atomic_bool m_cancelled{ false };
  bool m_failed = false;
  int64_t m_frames = 0;
  int64_t m_last_pts = 0;
};
