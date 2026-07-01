#pragma once

#include <memory>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libavformat/avformat.h>
}

struct avpacket_deleter {
  void operator()(AVPacket* pkt) const noexcept {
    if (pkt) {
      av_packet_free(&pkt);
    }
  }
};

using avframe_ptr = std::shared_ptr<AVFrame>;

struct avframe_deleter {
  void operator()(AVFrame* f) const noexcept {
    if (f) av_frame_free(&f);
  }
};

struct avformat_context_deleter {
  void operator()(AVFormatContext* ctx) const {
    if (ctx)
      avformat_close_input(&ctx);
  }
};

struct avcodec_context_deleter {
  void operator()(AVCodecContext* ctx) const noexcept {
    if (ctx) {
      avcodec_free_context(&ctx);
    }
  }
};
