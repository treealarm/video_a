#include "roi_stream_client.h"

#include <grpcpp/grpcpp.h>

#include "logging.h"
#include "roi_transcode.grpc.pb.h"

namespace rt = roi::transcode;

namespace {

bool encoder_from_name(const std::string& name, rt::Encoder& out)
{
  if (name == "auto")  { out = rt::ENCODER_AUTO;  return true; }
  if (name == "qsv")   { out = rt::ENCODER_QSV;   return true; }
  if (name == "vaapi") { out = rt::ENCODER_VAAPI; return true; }
  if (name == "nvenc") { out = rt::ENCODER_NVENC; return true; }
  if (name == "cpu")   { out = rt::ENCODER_CPU;   return true; }
  return false;
}

bool codec_from_name(const std::string& name, rt::Codec& out)
{
  if (name == "h265" || name == "hevc") { out = rt::CODEC_H265; return true; }
  if (name == "h264")                   { out = rt::CODEC_H264; return true; }
  if (name == "same")                   { out = rt::CODEC_SAME_AS_SOURCE; return true; }
  return false;
}

bool preset_from_name(const std::string& name, rt::Preset& out)
{
  if (name == "balanced") { out = rt::PRESET_BALANCED; return true; }
  if (name == "fastest")  { out = rt::PRESET_FASTEST;  return true; }
  if (name == "fast")     { out = rt::PRESET_FAST;     return true; }
  if (name == "slow")     { out = rt::PRESET_SLOW;     return true; }
  if (name == "slowest")  { out = rt::PRESET_SLOWEST;  return true; }
  return false;
}

}  // namespace

struct roi_stream_client::impl {
  std::shared_ptr<grpc::Channel> channel;
  std::unique_ptr<rt::RoiStreamTranscode::Stub> stub;
  grpc::ClientContext ctx;
  std::unique_ptr<grpc::ClientReaderWriter<rt::EncodeStreamRequest, rt::EncodeStreamResponse>>
    stream;
  bool open = false;
};

roi_stream_client::roi_stream_client()
  : m_impl(std::make_unique<impl>())
{
}

roi_stream_client::~roi_stream_client() = default;

bool roi_stream_client::open(const roi_encode_settings& settings, int32_t width, int32_t height,
  int32_t time_base_num, int32_t time_base_den, int32_t fps_num, int32_t fps_den,
  int32_t pix_fmt, std::string& error)
{
  rt::Encoder encoder{};
  rt::Codec codec{};
  rt::Preset preset{};
  if (!encoder_from_name(settings.encoder, encoder))
  {
    error = "unknown encoder '" + settings.encoder + "'";
    return false;
  }
  if (!codec_from_name(settings.codec, codec))
  {
    error = "unknown codec '" + settings.codec + "'";
    return false;
  }
  if (!preset_from_name(settings.preset, preset))
  {
    error = "unknown preset '" + settings.preset + "'";
    return false;
  }
  if (settings.output_path.empty())
  {
    error = "output_path is required: without it the service streams packets back instead of "
            "muxing a file, and nothing here would write them";
    return false;
  }

  // A raw frame of any size must fit: the default 4 MB limit rejects 1080p YUV420P (3.1 MB) only
  // just, and anything larger outright. The failure is a stream that dies mid-pass with a message
  // about the message size, which is worth not discovering on a long clip.
  grpc::ChannelArguments args;
  args.SetMaxSendMessageSize(-1);
  args.SetMaxReceiveMessageSize(-1);
  m_impl->channel = grpc::CreateCustomChannel(
    settings.target, grpc::InsecureChannelCredentials(), args);
  m_impl->stub = rt::RoiStreamTranscode::NewStub(m_impl->channel);
  m_impl->stream = m_impl->stub->EncodeStream(&m_impl->ctx);

  rt::EncodeStreamRequest msg;
  auto* open_msg = msg.mutable_open();
  open_msg->set_width(width);
  open_msg->set_height(height);
  open_msg->set_time_base_num(time_base_num);
  open_msg->set_time_base_den(time_base_den);
  open_msg->set_fps_num(fps_num);
  open_msg->set_fps_den(fps_den);
  open_msg->set_codec(rt::INPUT_CODEC_RAW);
  open_msg->set_pix_fmt(pix_fmt);
  open_msg->set_output_path(settings.output_path);
  if (!settings.movflags.empty())
    open_msg->set_movflags(settings.movflags);

  auto* s = open_msg->mutable_settings();
  s->set_encoder(encoder);
  s->set_codec(codec);
  s->set_preset(preset);
  if (settings.crf > 0) s->set_crf(settings.crf);
  if (settings.gop > 0) s->set_gop(settings.gop);

  auto* q = s->mutable_roi();
  if (settings.has_background_qp_delta)
    q->set_background_qp_delta(settings.background_qp_delta);
  if (settings.max_regions > 0)
    q->set_max_regions(settings.max_regions);
  if (settings.pad > 0)
    q->set_pad(settings.pad);
  if (settings.has_max_protected_fraction)
    q->set_max_protected_fraction(settings.max_protected_fraction);
  if (settings.margin_mode == "fraction")
    q->set_margin_mode(rt::MARGIN_MODE_FRACTION_OF_FRAME);
  else if (settings.margin_mode == "fixed" || !settings.margin_mode.empty())
    q->set_margin_mode(rt::MARGIN_MODE_FIXED_PIXELS);
  if (settings.has_margin_px)
    q->set_margin_px(settings.margin_px);
  if (settings.has_min_roi_side)
    q->set_min_roi_side(settings.min_roi_side);
  for (const auto& k : settings.kinds)
  {
    auto* kq = q->add_kinds();
    kq->set_kind(k.kind);
    kq->set_qp_delta(k.qp_delta);
    if (k.pad > 0)
      kq->set_pad(k.pad);
  }

  if (settings.has_max_qp_delta_per_frame)
  {
    auto* t = s->mutable_temporal_roi();
    t->set_max_qp_delta_per_frame(settings.max_qp_delta_per_frame);
  }

  if (settings.target_bitrate > 0)
  {
    auto* rc = s->mutable_rate_control();
    rc->set_target_bitrate(settings.target_bitrate);
    if (settings.max_average_overshoot > 0)
      rc->set_max_average_overshoot(settings.max_average_overshoot);
    if (settings.has_critical_importance)
      rc->set_critical_importance(settings.critical_importance);
  }

  if (!m_impl->stream->Write(msg))
  {
    // A rejected open closes the stream, and the reason is in the final status rather than in the
    // failed write itself.
    const auto status = m_impl->stream->Finish();
    error = status.ok() ? "the service closed the stream on open"
                        : status.error_message();
    return false;
  }

  m_impl->open = true;
  return true;
}

bool roi_stream_client::send_frame(int64_t pts, const uint8_t* data, size_t size,
  const std::vector<roi_box>& boxes)
{
  if (!m_impl->open)
    return false;

  rt::EncodeStreamRequest msg;
  auto* frame = msg.mutable_frame();
  frame->set_pts(pts);
  frame->set_data(data, size);

  if (!boxes.empty())
  {
    auto* list = frame->mutable_boxes();
    for (const auto& b : boxes)
    {
      auto* out = list->add_boxes();
      out->set_kind(b.kind);
      out->set_x(b.x);
      out->set_y(b.y);
      out->set_width(b.width);
      out->set_height(b.height);
      if (b.track_id > 0)
        out->set_track_id(b.track_id);
      out->set_confidence(b.confidence);
      for (const auto& p : b.polygon)
      {
        auto* pt = out->add_polygon();
        pt->set_x(p.x);
        pt->set_y(p.y);
      }
    }
  }

  return m_impl->stream->Write(msg);
}

bool roi_stream_client::send_frame_mask(int64_t pts, const uint8_t* data, size_t size,
  const uint8_t* mask, int32_t mask_width, int32_t mask_height)
{
  if (!m_impl->open)
    return false;

  rt::EncodeStreamRequest msg;
  auto* frame = msg.mutable_frame();
  frame->set_pts(pts);
  frame->set_data(data, size);

  auto* m = frame->mutable_mask();
  m->set_width(mask_width);
  m->set_height(mask_height);
  m->set_data(mask, static_cast<size_t>(mask_width) * mask_height);

  return m_impl->stream->Write(msg);
}

bool roi_stream_client::send_frame_qp_map(int64_t pts, const uint8_t* data, size_t size,
  int32_t block, int32_t cols, int32_t rows, int32_t width, int32_t height,
  const int8_t* deltas, size_t deltas_size)
{
  if (!m_impl->open)
    return false;

  rt::EncodeStreamRequest msg;
  auto* frame = msg.mutable_frame();
  frame->set_pts(pts);
  frame->set_data(data, size);

  auto* m = frame->mutable_qp_map();
  m->set_block(block);
  m->set_cols(cols);
  m->set_rows(rows);
  m->set_width(width);
  m->set_height(height);
  m->set_deltas(deltas, deltas_size);

  return m_impl->stream->Write(msg);
}

bool roi_stream_client::finish(roi_encode_report& report, std::string& error)
{
  if (!m_impl->open)
  {
    error = "stream was never opened";
    return false;
  }
  m_impl->open = false;

  m_impl->stream->WritesDone();

  rt::EncodeStreamResponse resp;
  while (m_impl->stream->Read(&resp))
  {
    if (resp.has_done())
    {
      report.frames = resp.done().frames();
      report.packets = resp.done().packets();
      report.bytes = resp.done().bytes();
      report.regions_dropped = resp.done().regions_dropped();
    }
  }

  const auto status = m_impl->stream->Finish();
  if (!status.ok())
  {
    error = status.error_message();
    return false;
  }
  return true;
}
