#include "frame_sampler.h"
#include "logging.h"

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

frame_sampler::frame_sampler(std::function<void(const decoded_frame&)> callback)
  : m_callback(std::move(callback))
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

  // Never hand non-key packets to the decoder at all.
  if (!(pkt->packet->flags & AV_PKT_FLAG_KEY)) return;

  // Enqueue only — the heavy decode + inference happens on the worker thread so this (the RTSP read
  // loop) returns immediately and keeps draining the socket. Drop the oldest queued keyframe when
  // the worker is behind: the newest keyframe is what matters, and blocking here would stall the
  // source (see the header note on live-video stutter).
  {
    std::scoped_lock lock(m_queue_mutex);
    while (m_queue.size() >= k_max_queue)
      m_queue.pop_front();
    m_queue.push_back(pkt);
  }
  m_cv.notify_one();
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
