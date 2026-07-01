#include "frame_sampler.h"
#include "logging.h"

frame_sampler::frame_sampler(std::function<void(const decoded_frame&)> callback)
  : m_callback(std::move(callback))
{
}

frame_sampler::~frame_sampler() = default;

void frame_sampler::on_packet(const std::shared_ptr<media_packet>& pkt)
{
  if (!pkt || !pkt->packet) return;
  if (pkt->media_type != AVMEDIA_TYPE_VIDEO) return;

  // Never hand non-key packets to the decoder at all.
  if (!(pkt->packet->flags & AV_PKT_FLAG_KEY)) return;

  std::scoped_lock lock(m_mutex);

  if (!ensure_decoder(*pkt)) return;

  if (avcodec_send_packet(m_decoder_ctx.get(), pkt->packet.get()) < 0)
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

  m_sws_ctx.reset(sws_getContext(
    width, height, format,
    width, height, AV_PIX_FMT_BGR24,
    SWS_BILINEAR, nullptr, nullptr, nullptr));

  if (!m_sws_ctx) return false;

  m_sws_width = width;
  m_sws_height = height;
  m_sws_format = format;
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
