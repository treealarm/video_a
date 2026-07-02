#include "crop_writer.h"
#include "inference/frame_sampler.h"
#include "logging.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <memory>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}

namespace fs = std::filesystem;

namespace {
struct avcodec_context_deleter_local {
  void operator()(AVCodecContext* ctx) const noexcept { if (ctx) avcodec_free_context(&ctx); }
};
struct sws_context_deleter_local {
  void operator()(SwsContext* ctx) const noexcept { if (ctx) sws_freeContext(ctx); }
};
struct avframe_deleter_local {
  void operator()(AVFrame* f) const noexcept { if (f) av_frame_free(&f); }
};
struct avpacket_deleter_local {
  void operator()(AVPacket* p) const noexcept { if (p) av_packet_free(&p); }
};
}

std::vector<uint8_t> encode_crop_jpeg(const decoded_frame& crop)
{
  if (crop.width <= 0 || crop.height <= 0 || crop.bgr.empty())
    return {};

  const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
  if (!codec)
  {
    log()->warn("crop_writer: no MJPEG encoder available");
    return {};
  }

  std::unique_ptr<AVCodecContext, avcodec_context_deleter_local> enc_ctx(avcodec_alloc_context3(codec));
  if (!enc_ctx) return {};

  enc_ctx->width = crop.width;
  enc_ctx->height = crop.height;
  enc_ctx->time_base = { 1, 25 };
  enc_ctx->pix_fmt = AV_PIX_FMT_YUVJ420P;

  if (avcodec_open2(enc_ctx.get(), codec, nullptr) < 0)
  {
    log()->warn("crop_writer: failed to open MJPEG encoder");
    return {};
  }

  std::unique_ptr<SwsContext, sws_context_deleter_local> sws(sws_getContext(
    crop.width, crop.height, AV_PIX_FMT_BGR24,
    crop.width, crop.height, AV_PIX_FMT_YUVJ420P,
    SWS_BILINEAR, nullptr, nullptr, nullptr));
  if (!sws) return {};

  std::unique_ptr<AVFrame, avframe_deleter_local> frame(av_frame_alloc());
  if (!frame) return {};
  frame->format = AV_PIX_FMT_YUVJ420P;
  frame->width = crop.width;
  frame->height = crop.height;
  if (av_frame_get_buffer(frame.get(), 32) < 0) return {};

  const uint8_t* src_data[1] = { crop.bgr.data() };
  int src_linesize[1] = { crop.width * 3 };
  sws_scale(sws.get(), src_data, src_linesize, 0, crop.height, frame->data, frame->linesize);

  if (avcodec_send_frame(enc_ctx.get(), frame.get()) < 0) return {};

  std::unique_ptr<AVPacket, avpacket_deleter_local> pkt(av_packet_alloc());
  if (!pkt) return {};

  if (avcodec_receive_packet(enc_ctx.get(), pkt.get()) < 0) return {};

  return std::vector<uint8_t>(pkt->data, pkt->data + pkt->size);
}

crop_writer::crop_writer(std::string base_path) : m_base_path(std::move(base_path)) {}

std::string crop_writer::write_crop(const std::string& watch_id, int64_t track_id, const decoded_frame& crop) const
{
  const auto jpeg = encode_crop_jpeg(crop);
  if (jpeg.empty())
    return {};

  const auto now_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm tm_buf{};
  gmtime_r(&now_t, &tm_buf);

  char date_dir[32];
  std::strftime(date_dir, sizeof(date_dir), "%Y/%m/%d/%H", &tm_buf);
  char timestamp[32];
  std::strftime(timestamp, sizeof(timestamp), "%Y%m%dT%H%M%S", &tm_buf);

  char rel_buf[512];
  std::snprintf(rel_buf, sizeof(rel_buf), "%s/%s/%lld_%s.jpg",
    watch_id.c_str(), date_dir, static_cast<long long>(track_id), timestamp);
  const std::string relative_path = rel_buf;

  const fs::path full_path = fs::path(m_base_path) / relative_path;

  std::error_code ec;
  fs::create_directories(full_path.parent_path(), ec);
  if (ec)
  {
    log()->warn("crop_writer: failed to create directory '{}': {}", full_path.parent_path().string(), ec.message());
    return {};
  }

  std::ofstream out(full_path, std::ios::binary);
  if (!out)
  {
    log()->warn("crop_writer: failed to open '{}' for writing", full_path.string());
    return {};
  }
  out.write(reinterpret_cast<const char*>(jpeg.data()), static_cast<std::streamsize>(jpeg.size()));

  return relative_path;
}
