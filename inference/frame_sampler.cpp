#include "frame_sampler.h"
#include "logging.h"

#include <algorithm>
#include <vector>

namespace {
// Map deprecated JPEG-range pixel formats (YUVJ*) to their modern equivalents. Feeding a YUVJ
// format to sws_getContext logs "deprecated pixel format used" on every frame; the full range is
// signalled explicitly via sws_setColorspaceDetails instead.
AVPixelFormat normalize_pixel_format(AVPixelFormat fmt, bool& full_range)
{
  switch (fmt)
  {
    case AV_PIX_FMT_YUVJ420P: full_range = true; return AV_PIX_FMT_YUV420P;
    case AV_PIX_FMT_YUVJ422P: full_range = true; return AV_PIX_FMT_YUV422P;
    case AV_PIX_FMT_YUVJ444P: full_range = true; return AV_PIX_FMT_YUV444P;
    case AV_PIX_FMT_YUVJ440P: full_range = true; return AV_PIX_FMT_YUV440P;
    default:                  full_range = false; return fmt;
  }
}
}

frame_sampler::frame_sampler(std::function<void(const decoded_frame&)> callback, uint32_t sample_fps)
  : m_callback(std::move(callback))
  , m_emit_period(std::chrono::milliseconds(1000 / (sample_fps > 0 ? sample_fps : 1)))
{
  m_worker = std::thread([this] { worker_loop(); });
}

frame_sampler::~frame_sampler()
{
  m_running = false;
  m_cv.notify_all();
  if (m_worker.joinable())
    m_worker.join();
}

void frame_sampler::on_packet(const std::shared_ptr<media_packet>& pkt)
{
  if (!pkt || !pkt->packet) return;
  if (pkt->media_type != AVMEDIA_TYPE_VIDEO) return;

  const bool is_key = (pkt->packet->flags & AV_PKT_FLAG_KEY) != 0;

  // Keyframes are timed whatever the mode, because the measurement is what chooses the mode.
  if (is_key)
  {
    note_keyframe();
    update_mode();
  }

  if (!m_full_decode && !is_key)
    return;  // the cheap path: non-key packets never reach the decoder at all

  // Enqueue only — the heavy decode + inference happens on the worker thread so this (the RTSP read
  // loop) returns immediately and keeps draining the socket. Blocking here would stall the source
  // (see the header note on live-video stutter), so the queue sheds instead.
  {
    std::scoped_lock lock(m_queue_mutex);
    if (!m_full_decode)
    {
      while (m_queue.size() >= k_max_queue)
        m_queue.pop_front();
    }
    else if (m_queue.size() >= k_max_queue_full)
    {
      // Shed a whole GOP rather than a packet. Dropping one P-frame would leave the decoder
      // producing garbage until the next keyframe, which is worse than losing the second or two
      // this discards: pop the front, then keep popping until the queue starts on a keyframe
      // again — a packet the decoder can actually begin from.
      m_queue.pop_front();
      while (!m_queue.empty() && !(m_queue.front()->packet->flags & AV_PKT_FLAG_KEY))
        m_queue.pop_front();
      log()->warn("frame_sampler: decode is behind, dropped a GOP");
    }
    m_queue.push_back(pkt);
  }
  m_cv.notify_one();
}

void frame_sampler::note_keyframe()
{
  const auto now = std::chrono::steady_clock::now();
  m_keyframe_times.push_back(now);
  while (m_keyframe_times.size() > k_interval_gaps + 1)
    m_keyframe_times.pop_front();

  if (m_keyframe_times.size() < 2)
    return;

  // Median of the gaps, not the mean. The outliers here are one-sided and large -- a reconnect or
  // a stalled publisher contributes one enormous gap -- and a mean would let a single one of them
  // push a well-behaved stream onto the expensive path and keep it there.
  std::vector<std::chrono::milliseconds> gaps;
  gaps.reserve(m_keyframe_times.size() - 1);
  for (size_t i = 1; i < m_keyframe_times.size(); ++i)
    gaps.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(
      m_keyframe_times[i] - m_keyframe_times[i - 1]));
  std::sort(gaps.begin(), gaps.end());
  m_keyframe_interval = gaps[gaps.size() / 2];
}

void frame_sampler::update_mode()
{
  if (m_keyframe_interval.count() == 0)
    return;  // nothing measured yet; the cheap path is also the one we would have chosen

  // Hysteresis, because the interesting case is a stream whose keyframe spacing sits right at the
  // requested period: without a band it would flip modes on jitter alone, and each flip costs a
  // decoder that has to resynchronise.
  const auto enter = m_emit_period * 3 / 2;
  const auto leave = m_emit_period * 11 / 10;

  const bool want_full = m_full_decode ? (m_keyframe_interval > leave)
                                       : (m_keyframe_interval > enter);
  if (want_full == m_full_decode)
    return;

  m_full_decode = want_full;
  log()->info("frame_sampler: keyframe interval {} ms vs sample period {} ms — {} decode",
    m_keyframe_interval.count(), m_emit_period.count(),
    m_full_decode ? "switching to full" : "back to keyframe-only");
}

void frame_sampler::worker_loop()
{
  while (m_running)
  {
    std::shared_ptr<media_packet> pkt;
    {
      std::unique_lock lock(m_queue_mutex);
      m_cv.wait(lock, [&] { return !m_queue.empty() || !m_running; });
      if (!m_running)
        break;
      pkt = std::move(m_queue.front());
      m_queue.pop_front();
    }
    if (pkt)
      decode_packet(*pkt);
  }
}

void frame_sampler::decode_packet(const media_packet& pkt)
{
  if (!ensure_decoder(pkt)) return;

  if (avcodec_send_packet(m_decoder_ctx.get(), pkt.packet.get()) < 0)
    return;

  AVFrame* frame = av_frame_alloc();
  if (!frame) return;

  while (avcodec_receive_frame(m_decoder_ctx.get(), frame) == 0)
  {
    handle_decoded_frame(frame);
    av_frame_unref(frame);
  }
  av_frame_free(&frame);
}

bool frame_sampler::ensure_decoder(const media_packet& pkt)
{
  if (m_decoder_ctx) return true;

  if (!pkt.codec_parameters) return false;

  const AVCodec* codec = avcodec_find_decoder(pkt.codec_parameters->codec_id);
  if (!codec)
  {
    log()->error("frame_sampler: no decoder for codec_id={}", static_cast<int>(pkt.codec_parameters->codec_id));
    return false;
  }

  AVCodecContext* ctx = avcodec_alloc_context3(codec);
  if (!ctx) return false;

  if (avcodec_parameters_to_context(ctx, pkt.codec_parameters.get()) < 0)
  {
    avcodec_free_context(&ctx);
    return false;
  }

  if (avcodec_open2(ctx, codec, nullptr) < 0)
  {
    avcodec_free_context(&ctx);
    return false;
  }

  m_decoder_ctx.reset(ctx);
  return true;
}

bool frame_sampler::ensure_sws_context(int width, int height, AVPixelFormat format)
{
  if (m_sws_ctx && width == m_sws_width && height == m_sws_height && format == m_sws_format)
    return true;

  bool full_range = false;
  const AVPixelFormat src_format = normalize_pixel_format(format, full_range);

  m_sws_ctx.reset(sws_getContext(
    width, height, src_format,
    width, height, AV_PIX_FMT_BGR24,
    SWS_BILINEAR, nullptr, nullptr, nullptr));

  if (!m_sws_ctx) return false;

  if (full_range)
  {
    // JPEG full-range (0..255) input — tell sws so luma/chroma aren't wrongly rescaled.
    const int* coeffs = sws_getCoefficients(SWS_CS_ITU601);
    sws_setColorspaceDetails(m_sws_ctx.get(), coeffs, /*srcRange=*/1, coeffs, /*dstRange=*/1, 0, 1 << 16, 1 << 16);
  }

  m_sws_width = width;
  m_sws_height = height;
  m_sws_format = format; // cache on the ORIGINAL format so the comparison above still matches
  return true;
}

void frame_sampler::handle_decoded_frame(AVFrame* frame)
{
  if (!m_callback) return;
  if (frame->width <= 0 || frame->height <= 0) return;

  // The schedule, applied in both modes so that sample_fps means the same thing whichever one is
  // running -- in full decode it is what makes most decoded frames free, and in keyframe-only it
  // stops a stream that keys faster than asked from running inference more often than asked.
  //
  // Tested before the colour conversion below, which is the expensive part of this function.
  //
  // The tolerance matters more than it looks. A camera keying once a second against a one-second
  // period lands a few milliseconds early as often as late, and a bare `>= period` would reject
  // every early arrival and halve the rate to one frame every two seconds.
  const auto now = std::chrono::steady_clock::now();
  if (m_next_emit.time_since_epoch().count() == 0)
    m_next_emit = now;  // first frame is always due
  if (now < m_next_emit - m_emit_period / 10)
    return;
  // Advanced from the deadline rather than from this frame, so accepting an early one does not
  // pull the whole schedule forward with it. Measuring from the last emit instead made every
  // interval come out a tolerance short, which over a run is a tenth more inference than asked.
  m_next_emit += m_emit_period;
  if (m_next_emit < now)
    m_next_emit = now + m_emit_period;  // fell behind (a stall, a reconnect) -- resync rather than burst

  if (!ensure_sws_context(frame->width, frame->height, static_cast<AVPixelFormat>(frame->format)))
    return;

  decoded_frame out;
  out.width = frame->width;
  out.height = frame->height;
  out.bgr.resize(static_cast<size_t>(frame->width) * static_cast<size_t>(frame->height) * 3);
  out.captured_at = std::chrono::system_clock::now();

  uint8_t* dst_data[4] = { out.bgr.data(), nullptr, nullptr, nullptr };
  int dst_linesize[4] = { frame->width * 3, 0, 0, 0 };

  sws_scale(m_sws_ctx.get(), frame->data, frame->linesize, 0, frame->height, dst_data, dst_linesize);

  m_callback(out);
}
