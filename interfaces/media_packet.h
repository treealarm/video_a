#pragma once
#include <memory>

extern "C" {
#include <libavcodec/codec_par.h>
#include <libavcodec/packet.h>
#include <libavutil/avutil.h>
#include <libavutil/rational.h>
}

#include "av_deleters.h"

using avpacket_ptr = std::shared_ptr<AVPacket>;
using avframe_ptr = std::shared_ptr<AVFrame>;
using avcodec_parameters_ptr = std::shared_ptr<AVCodecParameters>;
using avformat_context_ptr = std::unique_ptr<AVFormatContext, avformat_context_deleter>;

struct media_packet {
  avpacket_ptr packet;
  int stream_index;
  AVRational time_base;
  AVMediaType media_type = AVMEDIA_TYPE_UNKNOWN;
  AVRational frame_rate{ 0, 1 };
  int sample_rate = 0;
  int nb_samples = 0;
  avcodec_parameters_ptr codec_parameters;
};

struct media_frame {
  avframe_ptr frame;
  int stream_index;
  AVRational time_base;
  AVMediaType media_type = AVMEDIA_TYPE_UNKNOWN;
};
