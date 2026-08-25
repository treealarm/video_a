#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
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

// Decodes keyframes only while that is enough to meet the requested rate, and everything when it
// is not.
//
// Keyframe-only is the cheap path and stays the default: an I-frame is self-contained, so one
// decoded frame costs one decoded frame, where decode-then-discard pays for the whole stream. Its
// price is that the sampling rate is then the *publisher's* keyframe interval, not ours. That is
// invisible on surveillance cameras, which key about once a second, and wrong on anything else: a
// file with a 8.3 s GOP yielded one detection per 8.3 s no matter what sample_fps asked for, and
// nothing in the system could say why.
//
// So the keyframe spacing is measured as packets arrive, and when it turns out to be longer than
// the requested sample period the sampler switches to decoding every packet and picks frames on
// its own schedule. Measured on this project's own footage, full decode of a 2688x1520 stream
// costs about 12% of one core — real, but affordable, and only paid by sources that need it.
//
// sample_fps is therefore an upper bound that is now actually enforced: the emit schedule applies
// in both modes, so a stream keying faster than the requested rate no longer runs inference more
// often than asked.
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
  // sample_fps is the target rate; 0 is read as 1. It bounds how often a decoded frame is handed
  // on, and decides which decode mode the measured keyframe spacing has to beat.
  frame_sampler(std::function<void(const decoded_frame&)> callback, uint32_t sample_fps);
  ~frame_sampler() override;

  void on_packet(const std::shared_ptr<media_packet>& pkt) override;

private:
  // Read-loop thread only.
  void note_keyframe();
  void update_mode();

  bool ensure_decoder(const media_packet& pkt);
  bool ensure_sws_context(int width, int height, AVPixelFormat format);
  void handle_decoded_frame(AVFrame* frame);
  void worker_loop();
  void decode_packet(const media_packet& pkt);

  std::function<void(const decoded_frame&)> m_callback;

  // Packets handed from the read loop to the decode/inference worker. The read loop must never
  // block, so the queue is bounded and sheds under pressure -- but how it sheds depends on the
  // mode, and the difference is not cosmetic.
  //
  // Keyframe-only: drop the oldest. A keyframe is worthless once a newer one exists, and each is
  // independently decodable, so dropping one costs exactly that frame.
  //
  // Full decode: dropping an arbitrary packet would break the reference chain and corrupt every
  // frame up to the next keyframe. Whole GOPs are discarded instead -- see on_packet -- so what
  // remains always starts on a packet the decoder can start from. The bound is correspondingly
  // larger, because one GOP is hundreds of packets rather than one.
  static constexpr size_t k_max_queue = 2;
  static constexpr size_t k_max_queue_full = 120;
  std::mutex m_queue_mutex;
  std::condition_variable m_cv;
  std::deque<std::shared_ptr<media_packet>> m_queue;
  std::atomic_bool m_running{true};
  std::thread m_worker;

  // How often a decoded frame may be handed on. Read-loop thread sets nothing here; the worker
  // thread owns m_next_emit, which is the deadline the schedule runs on.
  std::chrono::milliseconds m_emit_period{1000};
  std::chrono::steady_clock::time_point m_next_emit{};

  // Keyframe spacing and the mode it selects. Touched only by the read-loop thread, except
  // m_full_decode which the worker never reads either -- the mode decides what is enqueued, and
  // the worker simply decodes what it is given.
  static constexpr size_t k_interval_gaps = 5;
  std::deque<std::chrono::steady_clock::time_point> m_keyframe_times;
  std::chrono::milliseconds m_keyframe_interval{0};  // 0 = not measured yet
  bool m_full_decode = false;

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
