#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

#include "interfaces/media_sink.h"
#include "interfaces/av_deleters.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}

struct decoded_frame {
  std::vector<uint8_t> bgr;  // width * height * 3, BGR24, tightly packed
  int width = 0;
  int height = 0;
  std::chrono::system_clock::time_point captured_at;
};

// Keyframe-only decode: non-key packets are filtered out in on_packet() before ever reaching
// avcodec_send_packet — see the project plan's rationale (I-frames are self-contained, so
// skipping P/B packets entirely costs nothing on the decoder side, unlike decode-then-discard).
// Consequence: the effective sampling rate is bound by the stream's GOP/keyframe interval, not
// freely selectable — sample_fps in WatchRequest is a target/upper bound, not a guarantee.
//
// Decode + inference run on a dedicated worker thread, NOT in the caller's (rtsp_reader's) thread.
// This is load-bearing: on_packet is called from the RTSP read loop, so if the heavy work ran
// inline it would stop draining the socket for the duration of an inference pass. On an RTSP-over-
// TCP source (MediaMTX) that stalls the reader, back-pressures the server, and makes it deliver in
// bursts — which shows up as the *live* WebRTC view stuttering and fast-forwarding. Keeping the
// read loop free means the socket is always drained at real time; keyframes queue up and the worker
// drops the oldest if inference can't keep up (freshest-frame-wins, bounded latency).
class frame_sampler final : public media_sink {
public:
  explicit frame_sampler(std::function<void(const decoded_frame&)> callback);
  ~frame_sampler() override;

  void on_packet(const std::shared_ptr<media_packet>& pkt) override;

private:
  bool ensure_decoder(const media_packet& pkt);
  bool ensure_sws_context(int width, int height, AVPixelFormat format);
  void handle_decoded_frame(AVFrame* frame);
  void worker_loop();
  void decode_packet(const media_packet& pkt);

  std::function<void(const decoded_frame&)> m_callback;

  // Queue of keyframe packets handed from the read loop to the decode/inference worker. Bounded and
  // drop-oldest: a keyframe is worthless once a newer one exists, and the read loop must never block.
  static constexpr size_t k_max_queue = 2;
  std::mutex m_queue_mutex;
  std::condition_variable m_cv;
  std::deque<std::shared_ptr<media_packet>> m_queue;
  std::atomic_bool m_running{true};
  std::thread m_worker;

  // Decoder/sws state is touched only by the worker thread — no locking needed.
  using avcodec_context_ptr = std::unique_ptr<AVCodecContext, avcodec_context_deleter>;
  avcodec_context_ptr m_decoder_ctx;

  struct sws_context_deleter {
    void operator()(SwsContext* ctx) const noexcept { if (ctx) sws_freeContext(ctx); }
  };
  std::unique_ptr<SwsContext, sws_context_deleter> m_sws_ctx;
  int m_sws_width = 0;
  int m_sws_height = 0;
  AVPixelFormat m_sws_format = AV_PIX_FMT_NONE;
};
